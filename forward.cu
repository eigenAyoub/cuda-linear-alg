#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cudnn.h>

#include "backprop.cuh"
#include "utils.hpp"

const int IMAGE_SIZE  = 784;
const int NUM_IMAGES  = 60000;

const int INPUT_DIM  = 784;
const int HIDDEN_DIM = 10;
const int BATCH_SIZE = 64;

int main(){
    std::vector<float> X_train(NUM_IMAGES*IMAGE_SIZE), y_train(NUM_IMAGES);

    if (!read_mnist_data("data/train_mnist_images.bin",
                         "data/train_mnist_labels.bin",
                          X_train, 
                          y_train,
                          NUM_IMAGES,
                          IMAGE_SIZE
                        )) {
            return -1;
        }

    // first batch //
    std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);  // Batch_size (y)  x INPUT_DIM (x) >> [64, 784]
    std::vector<float> y_batch(BATCH_SIZE);              // Batch_size (y)  x INPUT_DIM (x) >> [64, 784]

    std::copy(X_train.begin(), X_train.begin() + BATCH_SIZE * INPUT_DIM, X_batch.begin());
    std::copy(y_train.begin(), y_train.begin() + BATCH_SIZE, y_batch.begin());

    std::vector<float> W1_h(INPUT_DIM*HIDDEN_DIM);
    std::vector<float> b1_h(HIDDEN_DIM);
    utils::xavier_init(W1_h.data(), b1_h.data(), INPUT_DIM, HIDDEN_DIM);

    float* X_train_d;
    float* y_train_d;

    //forward stuff
    float* W1_d;
    float* b1_d;
    float* Y1_d;  // Y1_h = X @ W1_h   >> [B, 10] >> [64x10]
    float* Z1_d;  // Z1_h = Y1_h + b1_h 
    float* smx;   // smx = log-softmax(Z1_h) ?? [B, 10]
    float* L;     // -logloss // [batch_size]
    float* l;     // sum(L)  // scalar

    cudaMalloc((void **) &X_train_d, sizeof(float)*X_batch.size());
    cudaMalloc((void **) &y_train_d, sizeof(float)*y_batch.size());

    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &smx, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &L, sizeof(float)*BATCH_SIZE);
    cudaMalloc((void **) &l, sizeof(float));

    cudaMemcpy(X_train_d, X_batch.data(), X_batch.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_train_d, y_batch.data(), y_batch.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);

    // backprop stuff
    float* dW1_d;
    float* db1_d;
    float* dZ1_d;
    float* dA1_d;

    cudaMalloc((void **) &dW1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &db1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &dZ1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &dA1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    dim3 blockDim(16,16);     // for [BATCH_SIZE, INPUT_DIM]
    dim3 gridDim(16,BATCH_SIZE/blockDim.y);

    dim3 blockDims1(10,32);   // for [BATCH_SIZE, HIDDEN_DIM]
    dim3 gridDims1(1,2);


    dim3 blockDim1D(1,32);    // for operations that operate on a long line ([BATCH_SIZE,1])
    dim3 gridDim1D(1,2);

    // forward pass
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
    coalesced_bias<<<gridDims1, blockDims1>>>(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
    softmax<<<gridDims1, blockDims1>>>(smx, Z1_d, HIDDEN_DIM); //  this is acually logsoftmax; not anymore
    logloss<<<2, 32>>>(L, smx, y_train_d, HIDDEN_DIM);  // we just pick the true label // L is [BATCH_SIZE,1]
    rLoss<<<2, 32>>>(l, L);

    // backward pass


    // 3. Add synchronization and kernel error check
    dA<<<gridDims1, blockDims1>>>(dA1_d, smx, y_train_d, HIDDEN_DIM);

    // quick buffer for dA A^T

    float* dA_AT; // dA @ A.T
    cudaMalloc((void **) &dA_AT, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    // mult <<<>>> careful it's not the order we want now.

    // dA =? A[row*hidden_unit+col] * (dA[row*hidden_unit+ col] - dA_AT[row] ) // done done 


    return 0;
}