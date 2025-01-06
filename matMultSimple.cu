#include <iostream>
#include <cuda_runtime.h>


__global__ void matMul(float *A, float *B, float *C, int N){
    int col = 0;
    int row = 0;
    C[row*N + col] = 0;
}


int main(){
    int N = 256*5;

    float A[N][N];
    float B[N][N];
    float C[N][N];

    int size = N*N*sizeof(float);
    
    // malloc
    float *A_d, *B_d, *C_d;

    cudaMalloc((void **) &A_d,size);
    cudaMalloc((void **) &B_d,size);
    cudaMalloc((void **) &C_d,size);

    // Memcpy

    cudaMemcpy(A_d,A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(B_d,B,size,cudaMemcpyHostToDevice);
    
    // computations:
    int numOfBlocks = 5;
    int threadsPerBlock = 256;

    matMul<<<numOfBlocks, threadsPerBlock>>>(A_d, B_d, C_d, N);

    cudaMemcpy(C,C_d,size,cudaMemcpyDeviceToHost);

    // moving data back


    // free Cuda


    return 0;
}