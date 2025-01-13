#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>

#include "backprop.cuh"
#include "utils.hpp"

const int IMAGE_SIZE  = 784;
const int NUM_IMAGES  = 60000;
//const int IMAGE_WIDTH = 28;

const int INPUT_DIM  = 784;
const int HIDDEN_DIM = 10;
//const int BATCH_SIZE = 1024;

void verify_multiplication() {
    // Setup test data
}

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

    float* X_train_d;
    float* y_train_d;

    std::cout << "Train size "  << X_train.size() << "\n";
    std::cout << "Labels size " << y_train.size() << "\n";

    cudaMalloc((void **) &X_train_d, sizeof(float)*X_train.size());
    cudaMalloc((void **) &y_train_d, sizeof(float)*y_train.size());

    cudaMemcpy(X_train_d, X_train.data(), X_train.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_train_d, y_train.data(), y_train.size()*sizeof(float), cudaMemcpyHostToDevice);


    std::vector<float> W1_h(INPUT_DIM*HIDDEN_DIM);
    std::vector<float> b1_h(HIDDEN_DIM);

    utils::xavier_init(W1_h.data(), b1_h.data(), INPUT_DIM, HIDDEN_DIM);


    float * W1_d;
    float * b1_d;
    float * Y1_d;  // Y1_h = X @ W1_h   >> [B, 10] >> [64x10]

    int BATCH_SIZE = 64;

    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, W1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(16,16); // data is [64 x 784]
    dim3 gridDim(49,4);

    utils::Timer mxx("Matrix computation took");
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, 64);

    mxx.report();


    std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);  // First batch only
    std::vector<float> Y_cpu(BATCH_SIZE * HIDDEN_DIM, 0.0f);
    std::vector<float> Y_gpu(BATCH_SIZE * HIDDEN_DIM);

    // Copy first batch from X_train
    std::copy(X_train.begin(), 
             X_train.begin() + BATCH_SIZE * INPUT_DIM, 
             X_batch.begin());

    // CPU multiplication
    for(int i = 0; i < BATCH_SIZE; i++) {
        for(int j = 0; j < HIDDEN_DIM; j++) {
            float sum = 0.0f;
            for(int k = 0; k < INPUT_DIM; k++) {
                sum += X_batch[i * INPUT_DIM + k] * W1_h[k * HIDDEN_DIM + j];
            }
            Y_cpu[i * HIDDEN_DIM + j] = sum;
        }
    }

    // Get GPU result
    cudaMemcpy(Y_gpu.data(), W1_d, 
               BATCH_SIZE * HIDDEN_DIM * sizeof(float), 
               cudaMemcpyDeviceToHost);
    // Print GPU matrix (64x10)

    std::cout << "First few elements comparison:\n";
    for(int i = 0; i < 5; i++) {
        std::cout << "CPU: " << Y_cpu[i] << " GPU: " << Y_gpu[i] 
                  << " diff: " << std::abs(Y_cpu[i] - Y_gpu[i]) << "\n";
    }

    /**
     * 
     * 

    int N = 1000; 
    float* h_A     = new float[N*N];

    srand(2025);
    for(int i = 0; i < N*N; ++i) {
        h_A[i] = static_cast<float>(rand() % 100) / 10.0f; // e.g. 0-9.9
    }


    utils::Timer smx("Softmax using CuDNN took > ");
     // 3. Apply Softmax to h_A
    gpuSoftmax(h_A, N*N);

    smx.report();

    cudaFree(X_train_d);
    cudaFree(y_train_d);

    //std::vector<float> Y1_h(BATCH_SIZE*HIDDEN_DIM); // Y1_h = X @ W1_h
    //std::vector<float> Z1_h(BATCH_SIZE*HIDDEN_DIM); // Z1_h = Y1_h + b1_h 
    //std::vector<float> A1_h(BATCH_SIZE*HIDDEN_DIM); // A1_h = relu(Z1_h)

    float * Z1_d;  // Z1_h = Y1_h + b1_h 
    float * A1_d;  // A1_h = relu(Z1_h) //or softmax
    float * smaxSumBuffer; // sum over batch size, could be useful for softmax later (sum per row)
    float * loss;  // loss =  Σ - q_i log(A_i)  // sum over samples?
    float * rloss; // sum over loss_i

    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &A1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, W1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);


    dim3 block(128,10); 
    dim3 grid(ceil(784/128), 1); 

    // for the first one: Y1_h = X @ W1_h 
    //int start_index = 0;
    //batch_multiply(X_train_d, W1_d, Y1_d, start_index, start_index+BATCH_SIZE, 784);
    //shared_bias(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
    //logloss<<<ceil(),BATCH_SIZE>>>(loss, A1_d);
    //rloss(rloss);

    // for softmax
    //dim3 blockSmx(BATCH_SIZE,10); 
    //dim3 gridSmx(1,1); 
    //softmax<<<blockSmx,gridSmx>>>(A1_d, Z1_d, smaxSumBuffer); 

    //dim3 numBx1d((BATCH_SIZE + tpb1d.x - 1) / tpb1d.x);
    //dim3 numBx2d(std::ceil(BATCH_SIZE/tpb2d.x), std::ceil(HIDDEN_DIM/tpb1d.y));

    //rSoftmax<<<numBlocks, tpbl>>>(S, true_y, rS, B, C);
     */





    return 0;
}