#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

const int IMAGE_SIZE  = 784;
const int NUM_IMAGES  = 60000;
const int IMAGE_WIDTH = 28;

const int INPUT_DIM  = 784;
const int OUTPUT_DIM = 10;
const int HIDDEM_DIM = 10;

const int BATCH_SIZE = 1024;

#define TILE_WIDTH 10

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
int main(){

    std::ifstream X_train_file("data/train_mnist_images.bin", std::ios::binary);
    std::ifstream y_train_file("data/train_mnist_labels.bin", std::ios::binary);

    if (!X_train_file || !y_train_file){
        std::cout << "Oupsie" << "\n";
        return -1;
    }

    std::vector<uint8_t> X_train_buff(NUM_IMAGES*IMAGE_SIZE);
    std::vector<uint8_t> y_train_buff(NUM_IMAGES);

    std::vector<float> X_train(NUM_IMAGES*IMAGE_SIZE);
    std::vector<float> y_train(NUM_IMAGES);

    // ignore multiplying by `sizeof(uint8_t)` as it's = 1
    X_train_file.read(reinterpret_cast<char *>(X_train_buff.data()), IMAGE_SIZE*NUM_IMAGES);
    y_train_file.read(reinterpret_cast<char *>(y_train_buff.data()), NUM_IMAGES);

    std::copy(X_train_buff.begin(), X_train_buff.end(), X_train.begin());
    std::copy(y_train_buff.begin(), y_train_buff.end(), y_train.begin());

    // moving data to the DRAM

    float *X_train_d;
    float *y_train_d;

    std::cout << "Train size "  << X_train.size() << "\n";
    std::cout << "Labels size " << y_train.size() << "\n";

    cudaMalloc((void **) &X_train_d, sizeof(float)*X_train.size());
    cudaMalloc((void **) &X_train_d, sizeof(float)*y_train.size());

    std::vector<float> W1_h(INPUT_DIM*HIDDEM_DIM);
    std::vector<float> b1_h(HIDDEM_DIM);
    //std::vector<float> Y1_h(BATCH_SIZE*HIDDEM_DIM); // Y1_h = X @ W1_h
    //std::vector<float> Z1_h(BATCH_SIZE*HIDDEM_DIM); // Z1_h = Y1_h + b1_h 
    //std::vector<float> A1_h(BATCH_SIZE*HIDDEM_DIM); // A1_h = relu(Z1_h)

    float * W1_d;
    float * b1_d;
    float * Y1_d;  // Y1_h = X @ W1_h
    float * Z1_d;  // Z1_h = Y1_h + b1_h 
    float * A1_d;  // A1_h = relu(Z1_h) //or softmax
    float * smaxSumBuffer; // sum over batch size, could be useful for softmax later (sum per row)
    float * loss;  // loss =  Σ - q_i log(A_i)  // sum over samples?
    float * rloss; // sum over loss_i

    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);
    cudaMalloc((void **) &A1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);

    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, W1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(128,10); 
    dim3 grid(ceil(784/128), 1); 

    // for the first one: Y1_h = X @ W1_h 
    int start_index = 0;
    batch_multiply(X_train_d, W1_d, Y1_d, start_index, start_index+BATCH_SIZE, 784);
    shared_bias(Z1_d, Y1_d, b1_d, HIDDEM_DIM);
    //logloss<<<ceil(),BATCH_SIZE>>>(loss, A1_d);
    //rloss(rloss);

    // for softmax
    dim3 blockSmx(BATCH_SIZE,10); 
    dim3 gridSmx(1,1); 
    softmax<<<blockSmx,gridSmx>>>(A1_d, Z1_d, smaxSumBuffer); 

    //dim3 numBx1d((BATCH_SIZE + tpb1d.x - 1) / tpb1d.x);
    //dim3 numBx2d(std::ceil(BATCH_SIZE/tpb2d.x), std::ceil(HIDDEM_DIM/tpb1d.y));

    //rSoftmax<<<numBlocks, tpbl>>>(S, true_y, rS, B, C);

    return 0;
}