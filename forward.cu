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
    int d11 = 0;
    int d22 = 0;

    if (d1 > 30){
        d11 = 30;
    }
    if (d2 > 10){
        d22 = 10;
    }
    std::vector<float> vBack(d1*d2);
    cudaMemcpy(vBack.data(), dev_var, sizeof(float)*d1*d2, cudaMemcpyDeviceToHost);

    std::cout << << i*d2+j << " ";

    std::cout << "\nVisual of input " << vName << "\n\n";
    for (int i=0; i < d11; i++){
        for (int j=0; j < d22; j++){
            std::cout << i*d2+j << " ";
            std::cout << vBack[i*d2+j] << " ";
        }
        std::cout <<"\n";
    }
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
    utils::xavier_init(W1_h.data(), b1_h.data(), INPUT_DIM, HIDDEN_DIM);


    std::cout << "w1_h  before \n";
    for (int i=0; i < 10; i++){
        for (int j=0; j < 5; j++){
            std::cout << W1_h[i*HIDDEN_DIM+j] << " ";
        }
        std::cout << "\n";
    }

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

    //back(INPUT_DIM, HIDDEN_DIM, W1_d);

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


    //dim3 blockDim1D(1,32);    // for operations that operate on a long line ([BATCH_SIZE,1])
    //dim3 gridDim1D(1,2);

    dim3 blockDimsIn(10,32);   // for [BATCH_SIZE, HIDDEN_DIM]
    dim3 gridDimsIn(1,ceil(INPUT_DIM/32));

    //dim3 blockDimsW(10,16);   // for [BATCH_SIZE, HIDDEN_DIM]
    //dim3 gridDimsW(1,49);

    // forward pass
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
    cudaDeviceSynchronize();

    coalesced_bias<<<gridDims1, blockDims1>>>(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
    cudaDeviceSynchronize();

    softmax<<<gridDims1, blockDims1>>>(smx, Z1_d, HIDDEN_DIM); //  this is acually softmax; not anymore
    cudaDeviceSynchronize();

    logloss<<<2, 32>>>(L, smx, y_train_d, HIDDEN_DIM);  // we just pick the true label // L is [BATCH_SIZE,1]
    cudaDeviceSynchronize();


    rLoss<<<2, 32>>>(l, L);
    cudaDeviceSynchronize();

    //// backward starts here: 
    back(BATCH_SIZE, HIDDEN_DIM, smx, "dZ");

    dA<<<gridDims1, blockDims1>>>(dA1_d, smx, y_train_d, HIDDEN_DIM);
    cudaDeviceSynchronize();


    //// quick buffer for dA A^T
    dim3 blockDim44(16,16);     // for [BATCH_SIZE, INPUT_DIM]
    dim3 gridDim44(4,4);
    float* dA_AT; // dA @ A.T
    cudaMalloc((void **) &dA_AT, sizeof(float)*BATCH_SIZE*BATCH_SIZE);
    mult_A_B_T<<<gridDim44,blockDim44>>>(dA1_d, smx, dA_AT, BATCH_SIZE, HIDDEN_DIM, BATCH_SIZE);
    cudaDeviceSynchronize();



    dZ<<<gridDims1,blockDims1>>>(dZ1_d, smx, dA1_d, dA_AT, HIDDEN_DIM);
    cudaDeviceSynchronize();


    mult_A_T_B<<<blockDimsIn,gridDimsIn>>>(X_train_d, dZ1_d, dW1_d, INPUT_DIM, BATCH_SIZE, HIDDEN_DIM);
    cudaDeviceSynchronize();


    db<<<HIDDEN_DIM/64,64>>>(db1_d, dZ1_d, HIDDEN_DIM);
    cudaDeviceSynchronize();

    //back(INPUT_DIM, HIDDEN_DIM, dW1_d, "dW1");
    back(1, HIDDEN_DIM, db1_d, "db1");


    // updates:
    //update1D<<<1,1>>>(b1_d, db1_d, HIDDEN_DIM);
    //update2D<<<gridDimsW,blockDimsW>>>(W1_d, dW1_d, INPUT_DIM, HIDDEN_DIM);

    //std::vector<float> w1Back(W1_h.size());
    //cudaMemcpy(w1Back.data(), W1_d, sizeof(float)*W1_h.size(), cudaMemcpyDeviceToHost);

    //std::cout <<"dw1 after the update \n"; // W1 is input_dim x hidden_dim
    //for (int i=0; i < 10; i++){
    //    for (int j=0; j < 5; j++){
    //        std::cout << dw1Back[i*HIDDEN_DIM+j] << " ";
    //    }
    //    std::cout << "\n";
    //}


    return 0;
}
