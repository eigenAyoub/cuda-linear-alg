#include <iostream>
#include "backprop.cuh"
#include <cmath>
#include <vector>
#include <fstream>
#include <cudnn.h>
#include "cuda_runtime.h"
//#include <math_constants.h>

#define TILE_WIDTH 16
#define BATCH_SIZE 64

// we would like the TILE_WIDTH to be the same as the block width.
// so far we assume that the matrix is squared N x N

__global__ void
softmax(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    __shared__ float buffPerBlock[32];
    __shared__ int   max[32];

    if (threadIdx.x == 0) {
        max[threadIdx.y] = 0;
        buffPerBlock[threadIdx.y] = 0.0f;
        for (int i = 1; i < hidden_dim; i++){
            if (Z[row*hidden_dim+i] > Z[row*hidden_dim + max[threadIdx.y]]) {
                max[threadIdx.y] = i;
            } 
        }
    }
    __syncthreads();


    Z[row*hidden_dim+col] = exp(Z[row*hidden_dim+col] - Z[row*hidden_dim + max[threadIdx.y]]);
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i =0; i < hidden_dim; i++){
            buffPerBlock[threadIdx.y] += Z[row*hidden_dim+i];
        }
    }
    __syncthreads();

    A[row*hidden_dim+col] = -__logf(fmaxf(Z[row*hidden_dim+col]/buffPerBlock[threadIdx.y],1e-30f));
}

__global__ void
shared_bias(float* Z, float* Y, float* b, int hidden_dim){

    int row = blockDim.y*blockIdx.y + threadIdx.y; 
    int col = blockDim.x*blockIdx.x + threadIdx.x; 

    extern __shared__ float bias[];

    int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (tid < blockDim.x) { bias[threadIdx.x] = b[col]; }
    __syncthreads();

    if (row < BATCH_SIZE && col < hidden_dim){
        Z[row*hidden_dim + col] = Y[row*hidden_dim + col] + bias[threadIdx.x];
    }
    __syncthreads();

}

__global__ void
coalesced_bias(float* Z, float* Y, float* b, int hidden_dim)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    Z[row * hidden_dim + col] = Y[row * hidden_dim + col] + b[col];
}

__global__ void
mult(float* A, float* B, float* C, int Ay, int cWidth, int Bx){ // cWidth as common width.
    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < Ay && tIdx+i < cWidth) ? A[row*cWidth + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < Bx && tIdy+i < cWidth) ? B[(tIdy+ i)*Bx + col] : 0.0f;
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

// next kernel should have as many threads as the BATCH_SIZE // and just 1D
__global__ void
logloss(float* L, float *A, float* y_train, int hidden_dim){
    // A is assumed to be [BATCH_SIZE x N_CLASSES] // assumed to be the log softmax.
    // L = [-log(loss)] // [BATCH_SIZE]

    int row = threadIdx.x + blockDim.x*blockIdx.x;
    if (row<BATCH_SIZE){
        L[row] = A[row*hidden_dim+(int)y_train[row]];
    }
}

__global__ void
rLoss(float *l, float* L){
    // L is of size Batch_size
    int row = threadIdx.x + blockDim.x*blockIdx.x;

    if (row == 0){
        printf("from gpu, l = %f \n", l[0]);
    }
    __syncthreads();

    for (int i = BATCH_SIZE/2; i > 0; i = i/2){
        if (row < i) {
            L[row] += L[row+i];
        }
    }
    __syncthreads();

    l[0] =  L[0];

    if (row == 0){
        printf("> after the sum > from gpu, l = %f \n", l[0]);
    }
    
}

__global__ void
relu(float* A, float* Z, int hidden_dim){

    int row = blockIdx.y*blockDim.y + threadIdx.y ;
    int col = blockIdx.x*blockDim.x + threadIdx.x ;

    A[row*hidden_dim+col]  = (Z[row*hidden_dim+col] > 0)? Z[0]:0;
}

bool read_mnist_data(
    const std::string& images_path,
    const std::string& labels_path,
    std::vector<float>& images,
    std::vector<float>& labels,
    const int num_images,
    const int image_size   
) {
    // Open files
    std::cout << num_images << image_size << "\n";

    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file || !labels_file) {
        std::cerr << "Error opening MNIST files" << std::endl;
        return false;
    }

    // Create temporary buffers
    std::vector<uint8_t> images_buff(num_images * image_size);
    std::vector<uint8_t> labels_buff(num_images);

    // Read binary data
    images_file.read(reinterpret_cast<char*>(images_buff.data()), 
                    num_images * image_size);
    labels_file.read(reinterpret_cast<char*>(labels_buff.data()), 
                    num_images);

    // Resize output vectors
    images.resize(num_images * image_size);
    labels.resize(num_images);

    // Convert to float
    std::copy(images_buff.begin(), images_buff.end(), images.begin());
    std::copy(labels_buff.begin(), labels_buff.end(), labels.begin());

    return true;
}


void gpuSoftmax(float* data, int batch_size, int hidden_dim) {
    // thanks Claude
    // Create cuDNN handle
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Allocate device memory
    float *d_data;
    size_t matrix_size = batch_size * hidden_dim * sizeof(float);
    cudaMalloc((void**)&d_data, matrix_size);
    cudaMemcpy(d_data, data, matrix_size, cudaMemcpyHostToDevice);

    // Create tensor descriptor for batch_size x hidden_dim matrix
    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    // NCHW: batch_size x hidden_dim x 1 x 1
    cudnnSetTensor4dDescriptor(
        data_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,    // N: number of images
        hidden_dim,    // C: number of channels (features)
        1,            // H: height
        1             // W: width
    );

    // Perform softmax
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(
        cudnn,
        CUDNN_SOFTMAX_LOG,
        CUDNN_SOFTMAX_MODE_CHANNEL,  // Softmax across hidden_dim
        &alpha,
        data_desc,
        d_data,
        &beta,
        data_desc,
        d_data
    );

    // Copy result back
    cudaMemcpy(data, d_data, matrix_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroy(cudnn);
}

// Usage:
// gpuSoftmax(data, BATCH_SIZE, HIDDEN_DIM);