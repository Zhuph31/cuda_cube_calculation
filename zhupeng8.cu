#include <cmath>
#include <iostream>
#include <sys/time.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

// #define NUM_BANKS 32
// #define LOG_NUM_BANKS 5
// #define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
int block_size = 1024;
int n_stream = 30;

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

struct ExecRecord {
  ExecRecord() {}
  void print() const {}
};

struct ExecRecords {
  double cpu_record;
  struct GPURecords {
  } gpu_records;
};

inline void check_kernel_err() {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Error: kernel invoke failed, %s\n",
            cudaGetErrorString(err));
    exit(-1);
  }
}

void print_cube(float ***cube, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        printf("%lf, ", cube[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}

void cpu_malloc_cube(float ****cube_ref, int n) {
  (*cube_ref) = (float ***)malloc(n * sizeof(float **));
  for (int i = 0; i < n; i++) {
    (*cube_ref)[i] = (float **)malloc(n * sizeof(float *));
    for (int j = 0; j < n; j++) {
      (*cube_ref)[i][j] = (float *)malloc(n * sizeof(float));
    }
  }
}

void gen_cube(float ***cube, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        cube[i][j][k] = (float)((i + j + k) % 10) * (float)1.1;
      }
    }
  }
}

void cpu_calculation(float ***input, float ***output, int n) {
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
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

float sum(float ***output, int n) {
  float sum = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        sum += output[i][j][k] * (((i + j + k) % 10) ? 1 : -1);
      }
    }
  }

  return sum;
}

int main(int argc, char *argv[]) {
  std::string n_str;
  int n;

  if ((argc != 2)) {
    std::cerr << "Error: wrong number of argument, specify one argument for "
                 "the dimension of the cube.\n";
    return -1;
  } else {
    n_str = argv[1];

    try {
      n = std::stoi(n_str);
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

  float cpu_cal_sum = sum(output, n);
  printf("cpu result sum:%lf\n", cpu_cal_sum);

  return 0;
}