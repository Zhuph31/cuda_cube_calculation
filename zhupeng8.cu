/*
* ECE1782 - W2024 - Lab 2 - Sample Code
* Sample Test Cases (sum)

n, result 
100,18295201.010496
200,147100808.124588
300,497296827.464880
400,1179763265.153962
500,2305380127.308517
600,3985027420.060339
700,6329585154.758305
800,9449933335.045414
*/

#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

/*You can use the following for any CUDA function that returns cudaError_t
 * type*/
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code == cudaSuccess)
    return;

  fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
  if (abort)
    exit(code);
}

/*Use the following to get a timestamp*/
double getTimeStamp() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_usec / 1000000 + tv.tv_sec;
}

__global__ void jacobiKernel(float *a, float *b, int n, int last_row,
                             int offset) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) / (n * n) + offset;
  int j = ((blockIdx.x * blockDim.x + threadIdx.x) / n) % n;
  int k = (blockIdx.x * blockDim.x + threadIdx.x) % n;

  if (i >= last_row || j >= n || k >= n)
    return;

  // float newVal = 0.0f;
  // if(i > 0) newVal += b[(i-1)*n*n + j*n + k];
  // if(i < n-1) newVal += b[(i+1)*n*n + j*n + k];
  // if(j > 0) newVal += b[i*n*n + (j-1)*n + k];
  // if(j < n-1) newVal += b[i*n*n + (j+1)*n + k];
  // if(k > 0) newVal += b[i*n*n + j*n + (k-1)];
  // if(k < n-1) newVal += b[i*n*n + j*n + (k+1)];

  // a[i*n*n + j*n + k] = 0.8f * newVal;
  a[i * n * n + j * n + k] =
      0.8f * (((i > 0) ? b[(i - 1) * n * n + j * n + k] : 0) +
              ((i < n - 1) ? b[(i + 1) * n * n + j * n + k] : 0) +
              ((j > 0) ? b[i * n * n + (j - 1) * n + k] : 0) +
              ((j < n - 1) ? b[i * n * n + (j + 1) * n + k] : 0) +
              ((k > 0) ? b[i * n * n + j * n + (k - 1)] : 0) +
              ((k < n - 1) ? b[i * n * n + j * n + (k + 1)] : 0));
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Error: wrong number of args\n");
    exit(1);
  }

  int n = atoi(argv[1]);
  size_t number_of_elements = ((size_t)n) * n * n;
  size_t bytes = number_of_elements * sizeof(float);
  int num_streams = std::max(std::min(n / 10, 50), 2) + 1;
  int rows_per_stream = ceil(n / (num_streams - 1.0f));
  int last_stream_rows = n - (num_streams - 2) * rows_per_stream;
  // printf("Number of streams: %d\n", num_streams);
  dim3 threadsPerBlock(1024);
  dim3 blocksPerGrid((rows_per_stream * n * n + 1023) / 1024);

  gpuErrchk(cudaDeviceReset());

  float *h_a, *h_b;
  cudaMallocHost(&h_a, bytes);
  cudaMallocHost(&h_b, bytes);

  // Initialize b_host
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        h_b[i * n * n + j * n + k] = (float)((i + j + k) % 10) * 1.1f;
      }
    }
  }

  cudaStream_t streams[num_streams];
  for (int i = 1; i < num_streams; i++) {
    cudaStreamCreate(&streams[i]);
  }

  cudaEvent_t events[num_streams];
  for (int i = 1; i < num_streams; i++) {
    cudaEventCreate(&events[i]);
  }

  //================= Timing Begins ========================
  double start_time = getTimeStamp();

  /*Device allocations are included in timing*/
  float *d_a, *d_b;
  cudaMalloc(&d_a, bytes);
  cudaMalloc(&d_b, bytes);

  cudaMemcpyAsync(d_b, h_b, (1 + rows_per_stream) * n * n * sizeof(float),
                  cudaMemcpyHostToDevice, streams[1]);
  cudaEventRecord(events[1], streams[1]);
  jacobiKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[1]>>>(
      d_a, d_b, n, rows_per_stream, 0);
  cudaMemcpyAsync(h_a, d_a, rows_per_stream * n * n * sizeof(float),
                  cudaMemcpyDeviceToHost, streams[1]);
  for (int i = 2; i < (num_streams - 1); i++) {
    int offset = (i - 1) * rows_per_stream;
    cudaMemcpyAsync(d_b + (offset + 1) * n * n, h_b + (offset + 1) * n * n,
                    rows_per_stream * n * n * sizeof(float),
                    cudaMemcpyHostToDevice, streams[i]);
    cudaEventRecord(events[i], streams[i]);
    cudaStreamWaitEvent(streams[i], events[i - 1], 0);
    jacobiKernel<<<blocksPerGrid, threadsPerBlock, 0, streams[i]>>>(
        d_a, d_b, n, i * rows_per_stream, offset);
    cudaMemcpyAsync(h_a + offset * n * n, d_a + offset * n * n,
                    rows_per_stream * n * n * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[i]);
  }
  if (last_stream_rows > 0) {
    int offset = (num_streams - 2) * rows_per_stream;
    cudaMemcpyAsync(d_b + (offset + 1) * n * n, h_b + (offset + 1) * n * n,
                    (last_stream_rows - 1) * n * n * sizeof(float),
                    cudaMemcpyHostToDevice, streams[num_streams - 1]);
    cudaStreamWaitEvent(streams[num_streams - 1], events[num_streams - 2], 0);
    jacobiKernel<<<blocksPerGrid, threadsPerBlock, 0,
                   streams[num_streams - 1]>>>(d_a, d_b, n, n, offset);
    cudaMemcpyAsync(h_a + offset * n * n, d_a + offset * n * n,
                    last_stream_rows * n * n * sizeof(float),
                    cudaMemcpyDeviceToHost, streams[num_streams - 1]);
  }
  cudaDeviceSynchronize();
  double end_time = getTimeStamp();
  //================= Timing Ends ========================    
  int total_time_ms = (int)ceil((end_time - start_time) * 1000);
  double sum = 0.0;
  // compute a at host side and check the result
  // float *a_host = (float *)malloc(bytes);
  // for(int i = 0; i < n; i++) {
  //     for(int j = 0; j < n; j++) {
  //         for(int k = 0; k < n; k++) {
  //             if (i > 0) a_host[i*n*n + j*n + k] += h_b[(i-1)*n*n + j*n + k];
  //             if (i < n-1) a_host[i*n*n + j*n + k] += h_b[(i+1)*n*n + j*n +
  //             k]; if (j > 0) a_host[i*n*n + j*n + k] += h_b[i*n*n + (j-1)*n +
  //             k]; if (j < n-1) a_host[i*n*n + j*n + k] += h_b[i*n*n + (j+1)*n
  //             + k]; if (k > 0) a_host[i*n*n + j*n + k] += h_b[i*n*n + j*n +
  //             (k-1)]; if (k < n-1) a_host[i*n*n + j*n + k] += h_b[i*n*n + j*n
  //             + (k+1)]; a_host[i*n*n + j*n + k] *= 0.8f; if (a_host[i*n*n +
  //             j*n + k] != h_a[i*n*n + j*n + k]) {
  //                 printf("Mismatch at %d %d %d: %f %f\n", i, j, k,
  //                 a_host[i*n*n + j*n + k], h_a[i*n*n + j*n + k]);
  //             }
  //         }
  //     }
  // }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {
        sum += h_a[i * n * n + j * n + k] * (((i + j + k) % 10) ? 1 : -1);
      }
    }
  }

  printf("%lf %d\n", sum, total_time_ms);

  for (int i = 1; i < num_streams; i++) {
    cudaStreamDestroy(streams[i]);
  }
  for (int i = 1; i < num_streams; i++) {
    cudaEventDestroy(events[i]);
  }
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFreeHost(h_a);
  cudaFreeHost(h_b);
  gpuErrchk(cudaDeviceReset());
  return 0;
}
