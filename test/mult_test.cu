#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Tile dimensions
#define TILE_WIDTH 16

// CUDA kernel for computing C = A * B^T.
// A is of dimension (Ay x cWidth), and B is of dimension (Bx x cWidth)
// (with B stored in row-major order). In our problem, we have:
//   A: 32 x 10, B: 32 x 10, so B^T is 10 x 32, and C becomes 32 x 32.

__global__ void diagonal(float *A, float *B, float *C, int Ay, int cWidth, int Bx) {

    // A > Ay, cWidth
    // B.T > cWidth, Bx  so B > Bx, cWidth
    // typically, Bx = Ay here (use case >>> dA @ A.T) >> cWidth = BATCH_SIZE

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    float interVal = 0 ;

    for (int i= 0; i < cWidth; i++){
        interVal += A[row*cWidth+i]*B[row*cWidth+i];
    }

    if (row < Ay){
        C[row] = interVal;
    }
}
__global__ void diag(float *A, float *B, float *C, int Ay, int cWidth, int Bx) {

    // return diagona vector of A @ B.T
    // B stored in row major format
    // no tiles needed bro


    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        interVal += A[row*cWidth+i]*B[col*cWidth+i];
    }

    if (row < Ay  && col < Bx){
        C[row] = interVal;
    }
}
__global__ void mult_transposeB(float *A, float *B, float *C, int Ay, int cWidth, int Bx) {

    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < Ay && tIdx+i < cWidth) ? A[row*cWidth + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < Bx && tIdy+i < cWidth) ? B[(tIdy+ i) + col*cWidth] : 0.0f;
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
        }
        __syncthreads();
    }

    if (row < Ay  && col < Bx){
        C[row*Bx + col] = interVal;
    }
}

// CPU reference implementation for C = A * B^T.
void cpuMultTransposeB(const float *A, const float *B, float *C,
                         int Ay, int cWidth, int Bx) {
    // Note: A is (Ay x cWidth), and B is (Bx x cWidth)
    // B^T is then (cWidth x Bx), so C ends up (Ay x Bx)
    for (int i = 0; i < Ay; i++) {
        for (int j = 0; j < Bx; j++) {
            float sum = 0;
            for (int k = 0; k < cWidth; k++) {
                // B^T[j][k] equals B[j][k] because B is stored row-major as (Bx x cWidth)
                sum += A[i * cWidth + k] * B[j * cWidth + k];
            }
            C[i * Bx + j] = sum;
        }
    }
}

int main(void) {
    // Matrix dimensions:
    // A is 32 x 10.
    // B is 32 x 10.
    // When computing C = A * B^T, B^T is 10 x 32, so C becomes 32 x 32.
    int Ay = 32;
    int cWidth = 10;
    int Bx = 32;   // Number of rows in original B.

    size_t sizeA = Ay * cWidth * sizeof(float);
    size_t sizeB = Bx * cWidth * sizeof(float);  // B: 32 x 10.
    size_t sizeC = Ay * Bx * sizeof(float);

    // Allocate host memory.
    float *h_A = (float *)malloc(sizeA);
    float *h_B = (float *)malloc(sizeB);
    float *h_C = (float *)malloc(sizeC);
    float *h_C_ref = (float *)malloc(sizeC);

    // Seed the random number generator.
    srand(17);

    // Initialize A and B with random values between 0 and 1.
    for (int i = 0; i < Ay * cWidth; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < Bx * cWidth; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy input matrices to device.
    cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    // Define grid and block dimensions.
    // The output matrix C is 32 x 32.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((Bx + TILE_WIDTH - 1) / TILE_WIDTH,
                 (Ay + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the kernel.
    mult_transposeB<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, Ay, cWidth, Bx);
    cudaDeviceSynchronize();

    // Copy the result matrix C back to host.
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Compute the CPU reference result.
    cpuMultTransposeB(h_A, h_B, h_C_ref, Ay, cWidth, Bx);

    // Verify and print a few elements from both results.
    printf("GPU result (first 5 elements of C):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", h_C[i * Bx + j]);
        }
        printf("\n");
    }

    printf("\nCPU reference result (first 5 elements of C):\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%8.4f ", h_C_ref[i * Bx + j]);
        }
        printf("\n");
    }

    // Optionally, you can compute the maximum difference to check correctness.
    float max_diff = 0.0f;
    for (int i = 0; i < Ay * Bx; i++) {
        float diff = fabs(h_C[i] - h_C_ref[i]);
        if (diff > max_diff)
            max_diff = diff;
    }
    printf("\nMaximum difference between GPU and CPU results: %f\n", max_diff);

    // Cleanup device and host memory.
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
