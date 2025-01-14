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
//const int IMAGE_WIDTH = 28;

const int INPUT_DIM  = 784;
const int HIDDEN_DIM = 10;
const int BATCH_SIZE = 64;

// Call after GPU multiplication
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
    std::copy(X_train.begin(), X_train.begin() + BATCH_SIZE * INPUT_DIM, X_batch.begin());

    std::vector<float> W1_h(INPUT_DIM*HIDDEN_DIM);
    std::vector<float> b1_h(HIDDEN_DIM);
    utils::xavier_init(W1_h.data(), b1_h.data(), INPUT_DIM, HIDDEN_DIM);

    float* X_train_d;
    //float* y_train_d;
    float* W1_d;
    float* b1_d;

    float* Y1_d;  // Y1_h = X @ W1_h   >> [B, 10] >> [64x10]
    float* Z1_d;  // Z1_h = Y1_h + b1_h 
    float* smx; // smx = softmax(Z1_h) ?? [B, 10]

    cudaMalloc((void **) &X_train_d, sizeof(float)*X_batch.size());
    //cudaMalloc((void **) &y_train_d, sizeof(float)*y_train.size());
    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &smx, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    cudaMemcpy(X_train_d, X_batch.data(), X_batch.size()*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(y_train_d, y_train.data(), y_train.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    
    // data is [64 x 784] = [y, x]
    // [64 x 784] = [16, 16]*[4,16]
    dim3 blockDim(16,16); 
    dim3 gridDim(16,4);

    utils::Timer mxx("Matrix computation took");
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
    mxx.report();


    dim3 blockDims1(10,32); 
    dim3 gridDims1(1,2);

    //size_t bSize = 8*sizeof(float);
    cudaDeviceSynchronize();  
    utils::Timer biasTime("Time to add add bias, simple");
    coalesced_bias<<<gridDims1, blockDims1>>>(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
    cudaDeviceSynchronize();  
    biasTime.report();

    // Z1_d =  x @ W +  b is ready  >> size:  [BATCH_SIZE x HIDDEN_DIM] (10)
     
    std::vector<float> Zback(BATCH_SIZE * HIDDEN_DIM);
    cudaMemcpy(Zback.data(), Z1_d, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);


    std::vector<float> cudnnSmx(BATCH_SIZE*HIDDEN_DIM);  

    for (int i = 0 ; i < 10; ++i){
        for (int j = 0 ; j < HIDDEN_DIM; ++j){
            std::cout << Zback[i*HIDDEN_DIM+j] << " ";
        } 
        std::cout << "\n";
    }

    std::copy(Zback.begin(), Zback.end(), cudnnSmx.begin());

    std::cout << "\n\n";
    gpuSoftmax(cudnnSmx.data(), BATCH_SIZE, HIDDEN_DIM);

    std::cout << "True cudnn softmax: \n";
    for (int i = 0 ; i < 10; ++i){
        for (int j = 0 ; j < HIDDEN_DIM; ++j){
            std::cout << std::fixed << std::setprecision(4) <<
             cudnnSmx[i*HIDDEN_DIM+j] << " ";
        } 
        std::cout << "\n";
    }

    utils::Timer smxx("\n Time to softmax it >  ");
    softmax<<<gridDims1, blockDims1>>>(smx, Z1_d, HIDDEN_DIM);
    cudaDeviceSynchronize();  
    smxx.report();


    std::vector<float> smxback(BATCH_SIZE * HIDDEN_DIM);
    cudaMemcpy(smxback.data(), smx, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);


    std::cout << "my softmaxyy \n\n" << "\n";
    for (int i = 0 ; i < 10; ++i){
        for (int j = 0 ; j < HIDDEN_DIM; ++j){
            std::cout << std::fixed << std::setprecision(4) <<
              smxback[i*HIDDEN_DIM+j] << " ";
        } 
        std::cout << "\n";
    }



    return 0;
}