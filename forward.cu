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

void back(int d1, int d2, float* dev_var, std::string vName){


    std::vector<float> vBack(d1*d2);
    cudaMemcpy(vBack.data(), dev_var, sizeof(float)*d1*d2, cudaMemcpyDeviceToHost);

    int x  = min(d1, 100);

    std::cout << "\n" << vName << " : \n";
    for (int i=0; i < d1; i++){
        for (int j=0; j < d2; j++){
            std::cout << vBack[i*d2+j] << " ";
        }
        std::cout <<"\n";
    }
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("Kernel error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
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
    std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);  // Batch_size (y)  x INPUT_DIM (x) >> [64, 784jj]
    std::vector<float> y_batch(BATCH_SIZE);              // Batch_size (y)  x INPUT_DIM (x) >> [64, 784]

    std::copy(X_train.begin(), X_train.begin() + BATCH_SIZE * INPUT_DIM, X_batch.begin());
    std::copy(y_train.begin(), y_train.begin() + BATCH_SIZE, y_batch.begin());

    std::vector<float> W1_h(INPUT_DIM*HIDDEN_DIM);
    std::vector<float> b1_h(HIDDEN_DIM);

    utils::loadWeights("W1.txt", W1_h,INPUT_DIM, HIDDEN_DIM);
    utils::loadBiases("b1.txt", b1_h, HIDDEN_DIM);

    for (int i = 0; i < 10; i++){
        for (int j = 0; j < 10; j++){
            std::cout << W1_h[i*HIDDEN_DIM+j] << " ";
        }
        std::cout << "\n";
    }


    float *X_train_d, *y_train_d;

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

    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);

    //back(INPUT_DIM, HIDDEN_DIM, W1_d);

    // backprop stuf, 
    float *dW1_d, *db1_d, *dZ1_d, *dA1_d;

    cudaMalloc((void **) &dW1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &db1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &dZ1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &dA1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    dim3 blockDim(16,16);     // for [BATCH_SIZE, INPUT_DIM]
    dim3 gridDim(1,4);

    dim3 blockDims1(10,32);   // for [BATCH_SIZE, HIDDEN_DIM]
    dim3 gridDims1(1,2);


    dim3 blockDimf(10,16);     // for [BATCH_SIZE, INPUT_DIM]
    dim3 gridDimf(1,49);

    dim3 blockDimW(16,16);     // for [BATCH_SIZE, INPUT_DIM]
    dim3 gridDimW(1,49);

    for (unsigned int batch = 0 ; batch < 10; batch++){

        CHECK_CUDA(cudaMemcpy(X_train_d, X_train.data()+batch*BATCH_SIZE*INPUT_DIM, X_batch.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(y_train_d, y_train.data()+batch*BATCH_SIZE, y_batch.size()*sizeof(float), cudaMemcpyHostToDevice));

        // forward pass
        mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        coalesced_bias<<<gridDims1, blockDims1>>>(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        softmax<<<gridDims1, blockDims1>>>(smx, Z1_d, HIDDEN_DIM); //  this is acually softmax; not anymore
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        logloss<<<2, 32>>>(L, smx, y_train_d, HIDDEN_DIM);  // we just pick the true label // L is [BATCH_SIZE,1]
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        rLoss<<<2, 32>>>(l, L);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        back(1,1,l, "loss per batch");

        //// backward starts here: 
        dZ<<<gridDims1,blockDims1>>>(dZ1_d, smx, y_train_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        mult_A_T_B<<<gridDimW, blockDimW>>>(X_train_d, dZ1_d, dW1_d, INPUT_DIM, BATCH_SIZE, HIDDEN_DIM);
        //cudaError_t err = cudaGetLastError();
        //if (err != cudaSuccess) {
        //    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        //    return;
        //}
        //CHECK_CUDA(cudaDeviceSynchronize());
        //CHECK_KERNEL();

        db<<<1,32>>>(db1_d, dZ1_d, HIDDEN_DIM);

        update1D<<<1,HIDDEN_DIM>>>(b1_d, db1_d, HIDDEN_DIM);
        update2D<<<gridDimf,blockDimf>>>(W1_d, dW1_d, INPUT_DIM, HIDDEN_DIM);

        //back(INPUT_DIM, HIDDEN_DIM, dW1_d, "dW1: ");
        //back(1, HIDDEN_DIM, b1_d, "b1: ");

    }

    return 0;
}
