#include <cstdio>
#include <vector>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

#define BLOCK_SIZE 128
#define NUM_ITERS 20   // repeat to average out noise / expose steady-state cost

template <typename T>
__global__ void sfu_test(T *x, T *y, size_t N, int iters){
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        T in = x[idx];
        T out = in;
        #pragma unroll 1
        for(int i = 0; i < iters; i++){
            out = sin(out + in);   // dependent chain, prevents CSE/removal
        }
        y[idx] = out;
    }
}

template <typename T>
void run_test(const char* label){
    const size_t N = 1 << 26;   // scale down a bit; NUM_ITERS adds the real work
    std::vector<T> x(N), y(N);
    for(size_t i = 0; i < N; i++){
        x[i] = std::fmod((T)i, (T)(2.0 * M_PI));  // keep args in a sane range
    }
    size_t bytes = sizeof(T) * N;
    T *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));

    const int NUM_BLOCKS = (int)((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm-up (first launch pays context/JIT costs)
    sfu_test<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_x, d_y, N, NUM_ITERS);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    sfu_test<T><<<NUM_BLOCKS, BLOCK_SIZE>>>(d_x, d_y, N, NUM_ITERS);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("%s: %.3f ms  (%.3f ns/element/iter)\n",
           label, ms, (ms * 1e6) / (double(N) * NUM_ITERS));

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main(){
    run_test<float>("float  sin");
    run_test<double>("double sin");
    return 0;
}
