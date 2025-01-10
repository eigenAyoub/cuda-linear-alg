#include <cstdio>   // For printf
#include <cstdlib>  // For rand, srand
#include "utils.hpp"

#define N 64
#define TILE_WIDTH 16
using namespace utils;

__global__ void
mult(float* A, float* B, float* C){
    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i< N; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < N && tIdx+i < N) ? A[row*N + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < N && tIdy+i < N) ? B[(tIdy+ i)*N + col] : 0.0f;
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
        }
        __syncthreads();
    }

    if (row<N && col < N){
        C[row*N + col] = interVal;
    }
}

void cpuMult(const float* A, const float* B, float* C, int n)
{
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < n; ++c) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[r * n + k] * B[k * n + c];
            }
            C[r * n + c] = sum;
        }
    }
}

int main()
{
    // 1. Allocate host arrays (float*)
    float* h_A     = new float[N*N];
    float* h_B     = new float[N*N];
    float* h_C_cpu = new float[N*N];
    float* h_C_gpu = new float[N*N];

    // 2. Initialize A and B with random values
    srand(2025);
    for(int i = 0; i < N*N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f; // e.g. 0-9.9
        h_B[i] = static_cast<float>(rand() % 100) / 10.0f;
    }

    Timer t("CPU my ass");
    // 3. CPU reference multiplication
    cpuMult(h_A, h_B, h_C_cpu, N);
    t.report();

    float *x_A;
    cudaMalloc((void**)&x_A, N*N*sizeof(float));

    Timer tgpu("Tiling on GPU: time for malloc + Memcpy + kernel exec + Memcpy");
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N*N*sizeof(float));
    cudaMalloc((void**)&d_B, N*N*sizeof(float));
    cudaMalloc((void**)&d_C, N*N*sizeof(float));

    // 5. Copy A, B to device
    cudaMemcpy(d_A, h_A, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*N*sizeof(float), cudaMemcpyHostToDevice);

    // 6. Setup block/grid and launch kernel
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + block.x - 1) / block.x,
              (N + block.y - 1) / block.y);

    mult<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // 7. Copy result back to host
    cudaMemcpy(h_C_gpu, d_C, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    tgpu.report();

    // 8. Compare GPU vs CPU
    bool match = true;
    for(int i = 0; i < N*N; ++i) {
        float diff = fabs(h_C_gpu[i] - h_C_cpu[i]);
        if (diff > 1e-3f) { 
            match = false;
            break;
        }
    }

    if (match) {
        printf("SUCCESS: CPU and GPU results match.\n");
    } else {
        printf("ERROR: CPU and GPU results do NOT match.\n");
    }

    // 9. Cleanup
    delete[] h_A;
    delete[] h_B;
    delete[] h_C_cpu;
    delete[] h_C_gpu;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
