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
const int BATCH_SIZE = 32;

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
    
    // data is [64 x 784] = [y, x]
    // [64 x 784] = [16, 16]*[4,16]
    dim3 blockDim(16,16); 
    dim3 gridDim(16,4);

    utils::Timer mxx("Matrix computation took");
    mult<<<gridDim, blockDim>>>(X_train_d, W1_d, Y1_d, BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
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

    float max_diff = 0.0f;
    std::cout << "\nComparing results (GPU vs CPU):\n";
    for(int i = 0; i < BATCH_SIZE; i++) {
        std::cout << "\nRow " << i << ":\n";
        for(int j = 0; j < HIDDEN_DIM; j++) {
            float gpu_val = Y_gpu[i * HIDDEN_DIM + j];
            float cpu_val = Y_cpu[i * HIDDEN_DIM + j];
            float diff = std::abs(gpu_val - cpu_val);
            max_diff = std::max(max_diff, diff);
            
            std::cout << std::fixed << std::setprecision(4) 
                    << "(" << gpu_val << ", " << cpu_val << ") ";
        }
    }

    std::cout << "\n\nMax difference: " << max_diff << std::endl;

    return 0;
}