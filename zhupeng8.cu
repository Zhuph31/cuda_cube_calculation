#include <cmath>
#include <iostream>
#include <stdarg.h>
#include <sys/time.h>
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
  for (uint64_t i = 0; i < n * n * n; ++i) {
    if (h_output[i] != d_output[i]) {
      printf("idx:%lu, %lf vs %lf\n", i, h_output[i], d_output[i]);
      exit(1);
    }
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

float *gpu_calculation(float ***input, uint64_t n) {
  uint64_t elements = n * n * n;

  float *output;
  cudaMallocHost((void **)&output, elements * sizeof(float),
                 cudaHostAllocWriteCombined);

  float *pinned_flat_cube;
  cudaMallocHost((void **)&pinned_flat_cube, elements * sizeof(float),
                 cudaHostAllocWriteCombined);
  flatten_cube(input, pinned_flat_cube, n);

  {
    TimeCost tc;
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, elements * sizeof(float));
    cudaMalloc((void **)&d_output, elements * sizeof(float));

    cudaMemcpy(d_input, pinned_flat_cube, elements * sizeof(float),
               cudaMemcpyHostToDevice);

    uint64_t grid_dim = (elements + block_dim - 1) / block_dim;
    printf("grid_dim:%lu, block_dim:%lu\n", grid_dim, block_dim);
    basic<<<grid_dim, block_dim>>>(d_input, d_output, n);
    check_kernel_err();

    cudaMemcpy(output, d_output, elements * sizeof(float),
               cudaMemcpyDeviceToHost);
    printf("basic solution cost:%d\n", int(std::ceil(tc.get_elapsed())));
  }

  return output;
}

void gpu_cal_compare(float ***input, float ***cpu_output, uint64_t n) {
  float *gpu_output = gpu_calculation(input, n);

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
  cpu_calculation(input, output, n);
  double cpu_cal_sum = sum_cube(output, n);
  printf("cpu result sum:%lf\n", cpu_cal_sum);

  gpu_cal_compare(input, output, n);

  return 0;
}