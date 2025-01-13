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
const int HIDDEN_DIM = 256;
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
    float * W1_d;
    float * b1_d;
    float * Y1_d;  // Y1_h = X @ W1_h   >> [B, 10] >> [64x10]


    cudaMalloc((void **) &X_train_d, sizeof(float)*X_batch.size());
    //cudaMalloc((void **) &y_train_d, sizeof(float)*y_train.size());
    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    cudaMemcpy(X_train_d, X_batch.data(), X_batch.size()*sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(y_train_d, y_train.data(), y_train.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, W1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    
    //std::vector<float> recover(BATCH_SIZE * HIDDEN_DIM, 0.0f);  // First batch only
    //cudaMemcpy(recover.data(), W1_d, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);

    //for(int i = 0; i < BATCH_SIZE; i++) {
    //    std::cout << "Row " << i << ": ";
    //    for(int j = 0; j < 28; j++) {
    //        std::cout << std::fixed << std::setprecision(4) 
    //                << W1_h[i * HIDDEN_DIM + j] << " >> "
    //                << recover[i * HIDDEN_DIM + j] << "\n";
    //    }
    //    std::cout << "\n";
    //}

    dim3 blockDim(16,16); // data is [64 x 784]
    dim3 gridDim(16,4);

    // x >> 16*16 = 256
    // y >> 16*4 = 64 

    utils::Timer mxx("Matrix computation took");
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, 784, 256);
    mxx.report();

    std::vector<float> Y_cpu(BATCH_SIZE * HIDDEN_DIM, 0.0f);
    std::vector<float> Y_gpu(BATCH_SIZE * HIDDEN_DIM);

    cudaMemcpy(Y_gpu.data(), Y1_d, BATCH_SIZE * HIDDEN_DIM * sizeof(float), cudaMemcpyDeviceToHost);

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

    std::cout << "First few elements comparison:\n";
    for(int i = 0; i < 64*2; i++) {
        std::cout << "CPU: " << Y_cpu[i] << " GPU: " << Y_gpu[i] << "\n";
//                  << " diff: " << std::abs(Y_cpu[i] - Y_gpu[i]) << "\n";
    }



    /**
     * 
     * 

    for(int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "Row " << i << ": ";
        for(int j = 0; j < HIDDEN_DIM; j++) {
            std::cout << std::fixed << std::setprecision(4) 
                    << X_train[i * HIDDEN_DIM + j] << " ";
        }
        std::cout << "\n";
    }
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