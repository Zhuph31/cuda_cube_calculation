#include <cmath>
#include <iostream>
#include <stdarg.h>
#include <sys/time.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define FLAT_INDEX(array, i, j, k, n) (array[(i) * (n) * (n) + (j) * (n) + (k)])

uint64_t block_dim = 1024;
uint64_t n_stream = 30;

void debug_printf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}

class TimeCost {
  double get_timestamp() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000 + tv.tv_sec;
  }

  double start_ts;

public:
  TimeCost() { start_ts = get_timestamp(); }
  double get_elapsed() const { return get_timestamp() - start_ts; }
};

inline void check_kernel_err() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: kernel invoke failed, %s\n",
            cudaGetErrorString(err));
    exit(-1);
  }
}

void print_cube(float ***cube, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        printf("%lf, ", cube[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void cpu_malloc_cube(float ****cube_ref, uint64_t n) {
  (*cube_ref) = (float ***)malloc(n * sizeof(float **));
  for (uint64_t i = 0; i < n; i++) {
    (*cube_ref)[i] = (float **)malloc(n * sizeof(float *));
    for (uint64_t j = 0; j < n; j++) {
      (*cube_ref)[i][j] = (float *)malloc(n * sizeof(float));
    }
  }
}

void gen_cube(float ***cube, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        cube[i][j][k] = (float)((i + j + k) % 10) * (float)1.1;
      }
    }
  }
}

void cpu_calculation(float ***input, float ***output, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        float elem1 = i > 0 ? input[i - 1][j][k] : 0;
        float elem2 = i < n - 1 ? input[i + 1][j][k] : 0;
        float elem3 = j > 0 ? input[i][j - 1][k] : 0;
        float elem4 = j < n - 1 ? input[i][j + 1][k] : 0;
        float elem5 = k > 0 ? input[i][j][k - 1] : 0;
        float elem6 = k < n - 1 ? input[i][j][k + 1] : 0;

        output[i][j][k] =
            (float)0.8 * (elem1 + elem2 + elem3 + elem4 + elem5 + elem6);
      }
    }
  }
}

void flatten_cube(float ***cube, float *array, uint64_t n) {
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        array[i * n * n + j * n + k] = cube[i][j][k];
      }
    }
  }
}

__global__ void basic(const float *input, float *output, unsigned int n) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * blockDim.x;
  uint64_t global_thread_id = block_offset + thread_id;

  uint64_t i = global_thread_id / (n * n), j = global_thread_id % (n * n) / n,
           k = global_thread_id % (n * n) % n;

  if (i >= n || j >= n || k >= n) {
    return;
  }

  float elem1 = i > 0 ? FLAT_INDEX(input, i - 1, j, k, n) : 0;
  float elem2 = i < n - 1 ? FLAT_INDEX(input, i + 1, j, k, n) : 0;
  float elem3 = j > 0 ? FLAT_INDEX(input, i, j - 1, k, n) : 0;
  float elem4 = j < n - 1 ? FLAT_INDEX(input, i, j + 1, k, n) : 0;
  float elem5 = k > 0 ? FLAT_INDEX(input, i, j, k - 1, n) : 0;
  float elem6 = k < n - 1 ? FLAT_INDEX(input, i, j, k + 1, n) : 0;

  FLAT_INDEX(output, i, j, k, n) =
      (float)0.8 * (elem1 + elem2 + elem3 + elem4 + elem5 + elem6);

  // printf("global thread id:%lu, %lu,%lu,%lu, pos_elem:%lf, "
  //        "elems:%lf,%lf,%lf,%lf,%lf,%lf"
  //        ", res:%lf\n",
  //        global_thread_id, i, j, k, FLAT_INDEX(input, i, j, k, n), elem1,
  //        elem2, elem3, elem4, elem5, elem6, FLAT_INDEX(output, i, j, k, n));

  return;
}

double sum_cube(float ***output, uint64_t n) {
  double sum = 0;

  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t j = 0; j < n; j++) {
      for (uint64_t k = 0; k < n; k++) {
        sum += (double)output[i][j][k] * (((i + j + k) % 10) ? 1 : -1);
      }
    }
  }

  return sum;
}

void verify_result(float *h_output, float *d_output, uint64_t n) {
  uint64_t elements = n * n * n;
  int thread_num = elements / 10000;
  thread_num = thread_num == 0 ? 1 : thread_num;
  thread_num = thread_num > 10 ? 10 : thread_num;
  uint64_t elem_per_thread = (elements + thread_num - 1) / thread_num;
  // printf("using %d threads, elem per thread:%lu", thread_num,
  // elem_per_thread);

  std::vector<std::thread> threads;

  for (int i = 0; i < thread_num; ++i) {
    threads.emplace_back(std::thread([h_output, d_output, n, thread_id = i,
                                      elem_per_thread, elements]() {
      for (uint64_t pos = thread_id * elem_per_thread; pos < elements; ++pos) {
        if (h_output[pos] != d_output[pos]) {
          printf("idx:%lu, %lf vs %lf\n", pos, h_output[pos], d_output[pos]);
          exit(1);
        }
      }
    }));
  }

  for (int i = 0; i < thread_num; ++i) {
    threads[i].join();
  }
  printf("verified, equal\n");
}

void debug_host_array(float *h_array, int elements) {
  printf("debug host array\n");
  for (size_t i = 0; i < elements; ++i) {
    printf("%lf,", h_array[i]);
  }
  printf("\n");
}

void debug_device_array(float *d_array, int elements) {
  printf("debug device array\n");
  float *test = (float *)malloc(elements * sizeof(float));
  cudaMemcpy(test, d_array, elements * sizeof(float), cudaMemcpyDeviceToHost);
  debug_host_array(test, elements);
}

struct ExecRecord {
  double host_to_device_copy;
  double device_to_host_copy;
  double kernel_time;
  double total_time;
  void print() const {
    printf("%lf, %lf, %lf, %lf\n", total_time, host_to_device_copy, kernel_time,
           device_to_host_copy);
  }
};

// ==================== historic implementation ========================
float *basic_gpu(float ***input, uint64_t n, ExecRecord &record) {
  uint64_t elements = n * n * n;

  float *output;
  cudaMallocHost((void **)&output, elements * sizeof(float),
                 cudaHostAllocWriteCombined);

  float *pinned_flat_cube;
  cudaMallocHost((void **)&pinned_flat_cube, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  flatten_cube(input, pinned_flat_cube, n);

  {
    TimeCost total_tc;
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, elements * sizeof(float));
    cudaMalloc((void **)&d_output, elements * sizeof(float));

    TimeCost host_to_device_copy_tc;
    cudaMemcpy(d_input, pinned_flat_cube, elements * sizeof(float),
               cudaMemcpyHostToDevice);
    record.host_to_device_copy = host_to_device_copy_tc.get_elapsed();

    uint64_t grid_dim = (elements + block_dim - 1) / block_dim;
    // printf("grid_dim:%lu, block_dim:%lu\n", grid_dim, block_dim);

    TimeCost kernel_tc;
    basic<<<grid_dim, block_dim>>>(d_input, d_output, n);

    cudaDeviceSynchronize();
    check_kernel_err();
    record.kernel_time = kernel_tc.get_elapsed();

    TimeCost device_to_host_copy_tc;
    cudaMemcpy(output, d_output, elements * sizeof(float),
               cudaMemcpyDeviceToHost);
    record.device_to_host_copy = device_to_host_copy_tc.get_elapsed();

    cudaDeviceSynchronize();
    record.total_time = total_tc.get_elapsed();
  }

  return output;
}
// ==================== historic implementation ========================

float *gpu_calculation(float ***input, uint64_t n, ExecRecord &record) {
  uint64_t elements = n * n * n;

  float *output;
  cudaMallocHost((void **)&output, elements * sizeof(float),
                 cudaHostAllocWriteCombined);

  float *pinned_flat_cube;
  cudaMallocHost((void **)&pinned_flat_cube, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  flatten_cube(input, pinned_flat_cube, n);

  cudaStream_t streams[n_stream + 1];

  cudaEvent_t h_to_d_copy_start[n_stream + 1], h_to_d_copy_end[n_stream + 1],
      d_to_h_copy_start[n_stream + 1], d_to_h_copy_end[n_stream + 1],
      kernel_start[n_stream + 1], kernel_end[n_stream + 1];

  for (int i = 1; i <= n_stream; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&h_to_d_copy_start[i]);
    cudaEventCreate(&h_to_d_copy_end[i]);
    cudaEventCreate(&d_to_h_copy_start[i]);
    cudaEventCreate(&d_to_h_copy_end[i]);
    cudaEventCreate(&kernel_start[i]);
    cudaEventCreate(&kernel_end[i]);
  }

  std::vector<uint64_t> stream_elems(n_stream + 1), stream_elem_offset,
      stream_bytes(n_stream + 1), stream_byte_offset(n_stream + 1);

  uint64_t elem_per_stream = (elements + n_stream - 1) / n_stream,
           last_stream_elem = elements - (n_stream - 1) * elem_per_stream;

  for (size_t i = 1; i <= n_stream; ++i) {
    stream_elem_offset[i] = elem_per_stream * (i - 1);
    stream_elems[i] =
        std::min(elem_per_stream, elements - stream_elem_offset[i]);
    stream_byte_offset[i] = stream_elem_offset[i] * sizeof(float);
    stream_bytes[i] = stream_elems[i] * sizeof(float);
  }

  TimeCost total_tc;
  float *d_input, *d_output;
  cudaMalloc((void **)&d_input, elements * sizeof(float));
  cudaMalloc((void **)&d_output, elements * sizeof(float));

  for (int i = 1; i <= n_stream; ++i) {
    cudaMemcpy(d_input, pinned_flat_cube, elements * sizeof(float),
               cudaMemcpyHostToDevice);

    uint64_t grid_dim = (elements + block_dim - 1) / block_dim;

    basic<<<grid_dim, block_dim>>>(d_input, d_output, n);
    cudaDeviceSynchronize();
    check_kernel_err();

    cudaMemcpy(output, d_output, elements * sizeof(float),
               cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
  }
  record.total_time = total_tc.get_elapsed();

  return output;
}

void gpu_cal_compare(float ***input, float ***cpu_output, uint64_t n) {
  ExecRecord record;
  float *gpu_output = gpu_calculation(input, n, record);
  record.print();

  float *flat_cpu_output = (float *)malloc(n * n * n * sizeof(float));
  flatten_cube(cpu_output, flat_cpu_output, n);
  verify_result(flat_cpu_output, gpu_output, n);
}

int main(int argc, char *argv[]) {
  std::string n_str;
  uint64_t n;

  if ((argc != 2)) {
    std::cerr << "Error: wrong number of argument, specify one argument for "
                 "the dimension of the cube.\n";
    return -1;
  } else {
    n_str = argv[1];

    try {
      n = std::stoull(n_str);
    } catch (std::exception &e) {
      std::cerr << "Error, failed to convert n to integer, error "
                   "message:"
                << e.what() << '\n';
      return -1;
    }
  }

  // printf("specified n:%d\n", n);

  float ***input, ***output;
  cpu_malloc_cube(&input, n);
  cpu_malloc_cube(&output, n);
  gen_cube(input, n);
  TimeCost cpu_tc;
  cpu_calculation(input, output, n);
  printf("cpu cost:%lf\n", cpu_tc.get_elapsed());
  double cpu_cal_sum = sum_cube(output, n);
  printf("cpu result sum:%lf\n", cpu_cal_sum);

  gpu_cal_compare(input, output, n);

  return 0;
}