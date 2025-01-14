#include <iostream>
#include "backprop.cuh"
#include <cmath>
#include <vector>
#include <fstream>
#include <cudnn.h>
#include "cuda_runtime.h"

#define TILE_WIDTH 16

//#define BATCH_SIZE 32
// we would like the TILE_WIDTH to be the same as the block width.
// so far we assume that the matrix is squared N x N



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

/** 
// Borowing ideas from Izzat El Hajj.
// https://www.youtube.com/@ielhajj

const int IMAGE_WIDTH = 28;
const int INPUT_DIM  = 784;
const int OUTPUT_DIM = 10;
const int HIDDEM_DIM = 10;


__global__ void
relu(float* A, float* Z, int B, int F){
    // takes pre-activations Z
    // computes the activation function  A = relu(Z)
    // B: batch size
    // F: Feature size 
    int tIdx = blockIdx.x*blockDim.x + threadIdx.x ;
    int tIdy = blockIdx.y*blockDim.y + threadIdx.y ;
    if (tIdx< B && tIdy < F){
        A[tIdx] = (Z[tIdx]>0) ? Z[tIdx] : 0;
    }
    
}

__global__ void
softmax(float* I, int *S, int width){
    // should compute the logits 
}

__global__ void
rSoftmax(float* S, int *Y, float* rS, int B){
    // takes softmax logits S ~ [B,10]
    // returns rS ~ [B] > for each sample S_i,  it select pred_i = S_i[true_y] 
    // and return -log(pred_i)
    int indx = blockIdx.x*blockDim.x + threadIdx.x ;

    if (indx < BATCH_SIZE){
        rS[indx] = -logf(S[Y[indx]]);
    }
}

__global__ void
reduced_sum(float* rS, float* loss, int B){
    // implement the reduction loss pattern
}

__global__ void
batch_multiply(
                        float* A, 
                        float* B, 
                        float* C, 
                        int batch_begin, 
                        int batch_end, 
                        int N
                    ){

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


__global__ void
shared_bias(float* Z, float* Y, float* b, int hidden_dim){

    int row = blockDim.y*blockIdx.y + threadIdx.y; 
    int col = blockDim.x*blockIdx.x + threadIdx.x; 
    int bDimx  = blockDim.x;

    __shared__ float bias[bDimx];

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
logloss(float* L, float *A, float* y_train, int batch_s, int batch_e){
    // A is assumed to be [BATCH_SIZE x N_CLASSES]
    // L = [-log(A_i)] // i is the true class  
    // [batch_s, batch_e] begining and ending of the batch.
    // I am assuming the block size here is exacly [BATCH_SIZE,1]
    int tid = threadIdx.x;
    int N = 10;
    if ((batch_s+tid) < 600000 && tid < BATCH_SIZE){
        int row = tid;
        int col = static_cast<int>(y_train[batch_s+tid]);
        L[tid] = -log(A[row*N+col])  ;
    }
}

__global__ void
rloss(float* loss_scalar, float *loss_vector){
    // A is assumed to be [BATCH_SIZE x N_CLASSES]
    // L = [-log(A_i)] // i is the true class  
    // [batch_s, batch_e] begining and ending of the batch.
    // I am assuming the block size here is exacly [BATCH_SIZE,1]

}


__global__ void
softmax(float* A, float *Z, float* buffer){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    int tid = threadIdx.x + blockDim.x*threadIdx.y;

    float eZ = exp(Z[row*BATCH_SIZE+col]);

    __shared__ float buffPerBlock[blockDim.y];
    if (threadIdx.x == 0) {
        buffPerBlock[threadIdx.y] = 0.0f;
    }
    __syncthreads();

    buffPerBlock[threadIdx.y] += eZ;
    __syncthreads(); 

    if (threadIdx.x == 0){
        buffer[row] += buffPerBlock[threadIdx.y];
    }
    __syncthreads(); 

    A[row*BATCH_SIZE+col] = eZ/buffer[row];
}
*/

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

void gpuSoftmax(float* data, int size) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    float *d_data;
    cudaMalloc((void**)&d_data, size * sizeof(float));
    cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    cudnnSetTensor4dDescriptor(data_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, size, 1, 1);

    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &alpha, data_desc, d_data, &beta, data_desc, d_data);

    cudaMemcpy(data, d_data, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_data);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroy(cudnn);
}

