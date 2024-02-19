#include <algorithm>
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
uint64_t n_stream = 5;

#define gpu_err_check(ans) gpu_err_check_impl((ans), __FILE__, __LINE__)
inline void gpu_err_check_impl(cudaError_t code, const char *file, int line,
                               bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %d %s %s:%d\n", code, cudaGetErrorString(code),
            file, line);
    if (abort) {
      fflush(stderr);
      exit(code);
    }
  }
}

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

__global__ void basic_streaming(const float *input, float *output,
                                unsigned int n, int stream_id,
                                uint64_t stream_elems, uint64_t stream_offset) {
  int block_id = blockIdx.x;
  int thread_id = threadIdx.x;
  int block_offset = block_id * blockDim.x;

  uint64_t global_thread_id = stream_offset + block_offset + thread_id;
  uint64_t inner_stream_thread_id = block_offset + thread_id;

  uint64_t i = global_thread_id / (n * n), j = global_thread_id % (n * n) / n,
           k = global_thread_id % (n * n) % n;

  if (inner_stream_thread_id >= stream_elems || i >= n || j >= n || k >= n) {
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
          printf("unequal\n");
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
  double host_to_device_copy = 0;
  double device_to_host_copy = 0;
  double kernel_time = 0;
  double total_time = 0;
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

  float *pinned_flat_input;
  cudaMallocHost((void **)&pinned_flat_input, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  flatten_cube(input, pinned_flat_input, n);

  {
    TimeCost total_tc;
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, elements * sizeof(float));
    cudaMalloc((void **)&d_output, elements * sizeof(float));

    TimeCost host_to_device_copy_tc;
    cudaMemcpy(d_input, pinned_flat_input, elements * sizeof(float),
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

  float *pinned_output;
  gpu_err_check(cudaMallocHost((void **)&pinned_output,
                               elements * sizeof(float),
                               cudaHostAllocWriteCombined));

  float *pinned_flat_input;
  gpu_err_check(cudaMallocHost((void **)&pinned_flat_input,
                               elements * sizeof(float),
                               cudaHostAllocWriteCombined));
  flatten_cube(input, pinned_flat_input, n);
  // debug_host_array(pinned_flat_input, elements);

  cudaStream_t streams[n_stream + 1];

  cudaEvent_t h_to_d_copy_start[n_stream + 1], h_to_d_copy_end[n_stream + 1],
      d_to_h_copy_start[n_stream + 1], d_to_h_copy_end[n_stream + 1],
      kernel_start[n_stream + 1], kernel_end[n_stream + 1];

  // create streams and events
  for (int i = 1; i <= n_stream; ++i) {
    cudaStreamCreate(&streams[i]);
    cudaEventCreate(&h_to_d_copy_start[i]);
    cudaEventCreate(&h_to_d_copy_end[i]);
    cudaEventCreate(&d_to_h_copy_start[i]);
    cudaEventCreate(&d_to_h_copy_end[i]);
    cudaEventCreate(&kernel_start[i]);
    cudaEventCreate(&kernel_end[i]);
  }

  // calculate each stream's elements and offset
  // each stream take at least a layer of elements
  std::vector<uint64_t> stream_elems(n_stream + 1),
      stream_elem_offset(n_stream + 1), stream_bytes(n_stream + 1),
      stream_byte_offset(n_stream + 1);
  uint64_t elem_per_stream = (elements + n_stream - 1) / n_stream;
  uint64_t elem_per_layer = n * n;
  elem_per_stream =
      std::ceil(double(elem_per_stream) / double(elem_per_layer)) *
      elem_per_layer;

  for (size_t i = 1; i <= n_stream; ++i) {
    stream_elem_offset[i] = elem_per_stream * (i - 1);
    if (stream_elem_offset[i] > elements) {
      stream_elem_offset[i] = elements;
    }
    stream_elems[i] =
        std::min(elem_per_stream, elements - stream_elem_offset[i]);
    stream_byte_offset[i] = stream_elem_offset[i] * sizeof(float);
    stream_bytes[i] = stream_elems[i] * sizeof(float);
  }

  // debug stream stats
  // for (size_t i = 1; i <= n_stream; ++i) {
  //   printf("stream:%lu, %lu,%lu,%lu,%lu\n", i, stream_elem_offset[i],
  //          stream_elems[i], stream_byte_offset[i], stream_bytes[i]);
  // }

  // printf("stats calculated\n");

  TimeCost total_tc;
  float *d_input, *d_output;
  gpu_err_check(cudaMalloc((void **)&d_input, elements * sizeof(float)));
  gpu_err_check(cudaMalloc((void **)&d_output, elements * sizeof(float)));

  // start all copy event
  for (int i = 1; i <= n_stream; ++i) {
    if (stream_elems[i] <= 0) {
      break;
    }
    gpu_err_check(cudaEventRecord(h_to_d_copy_start[i], streams[i]));
    gpu_err_check(cudaMemcpyAsync(&(d_input[stream_elem_offset[i]]),
                                  &(pinned_flat_input[stream_elem_offset[i]]),
                                  stream_bytes[i], cudaMemcpyHostToDevice,
                                  streams[i]));
    gpu_err_check(cudaEventRecord(h_to_d_copy_end[i], streams[i]));
  }

  // start kernel
  for (int i = 1; i <= n_stream; ++i) {
    if (stream_elems[i] <= 0) {
      break;
    }

    // wait for the copy on the next stream to finish first
    if (i < n_stream && stream_elems[i + 1] > 0) {
      cudaStreamWaitEvent(streams[i], h_to_d_copy_end[i + 1]);
    }

    uint64_t grid_dim = (stream_elems[i] + block_dim - 1) / block_dim;
    // printf("stream:%d, grid_dim:%lu\n", i, grid_dim);
    gpu_err_check(cudaEventRecord(kernel_start[i], streams[i]));
    basic_streaming<<<grid_dim, block_dim, 0, streams[i]>>>(
        d_input, d_output, n, i, stream_elems[i], stream_elem_offset[i]);
    gpu_err_check(cudaEventRecord(kernel_end[i], streams[i]));
  }

  // start all copy back event
  for (int i = 1; i <= n_stream; ++i) {
    if (stream_elems[i] <= 0) {
      break;
    }

    // wait for the kernel on the current stream to finish first
    cudaStreamWaitEvent(streams[i], kernel_end[i]);

    gpu_err_check(cudaEventRecord(d_to_h_copy_start[i], streams[i]));
    gpu_err_check(cudaMemcpyAsync(&(pinned_output[stream_elem_offset[i]]),
                                  &(d_output[stream_elem_offset[i]]),
                                  stream_bytes[i], cudaMemcpyDeviceToHost,
                                  streams[i]));
    gpu_err_check(cudaEventRecord(d_to_h_copy_end[i], streams[i]));
  }

  cudaDeviceSynchronize();

  // update record
  record.total_time = total_tc.get_elapsed();

  record.kernel_time = 0;
  record.device_to_host_copy = 0;

  for (int i = 1; i <= n_stream; ++i) {
    if (stream_elems[i] <= 0) {
      break;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, h_to_d_copy_start[i], h_to_d_copy_end[i]);
    // printf("stream:%d, htod copy time:%f\n", i, ms);
    record.host_to_device_copy += ms / 1000;
    cudaEventElapsedTime(&ms, kernel_start[i], kernel_end[i]);
    record.kernel_time += ms / 1000;
    cudaEventElapsedTime(&ms, d_to_h_copy_start[i], d_to_h_copy_end[i]);
    // printf("stream:%d, dtoh copy time:%f\n", i, ms);
    record.host_to_device_copy += ms / 1000;
    record.device_to_host_copy += ms / 1000;
  }

  return pinned_output;
}

void gpu_cal_compare(float ***input, float ***cpu_output, uint64_t n) {
  {
    ExecRecord record;
    basic_gpu(input, n, record);
    record.print();
  }

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