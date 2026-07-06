#include <cstdio>
#include <vector>
#include <cmath>

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
#define NUM_ITERS 20

template <typename T>
__global__ void sfu_test(T *x, T *y, size_t N, int iters){
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    if(idx < N){
        T in = x[idx];
        T out = in + (T)1.0;   // avoid 0 for sqrt/log/rsqrt on first pass

        #pragma unroll 1
        for(int i = 0; i < iters; i++){
            T a = sqrt(out);
            T b = exp((T)0.001 * out);      // scaled to avoid overflow across iters
            T c = sin(out);
            T d = cos(out);
            T e = log(fabs(out) + (T)1.0);  // guard against log(<=0)

            // combine so every term is live and the compiler can't drop any call
            out = a + b + c + d + e;
        }
        y[idx] = out;
    }
}

template <typename T>
void run_test(const char* label){
    const size_t N = 1 << 24;
    std::vector<T> x(N), y(N);
    for(size_t i = 0; i < N; i++){
        x[i] = std::fmod((T)i, (T)(2.0 * M_PI));
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
    int device = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    run_test<float>("float  mixed");
    run_test<double>("double mixed");
    return 0;
}
