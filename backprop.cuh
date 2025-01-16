#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

// Constants (if needed globally)
#define TILE_WIDTH 16

__global__ void shared_bias(float* Z, float* Y, float* b, int hidden_dim);
__global__ void coalesced_bias(float* Z, float* Y, float* b, int hidden_dim);

// forward functions 
__global__ void mult(float* A, float* B, float* C, int Ax, int cWidth, int By);
__global__ void softmax(float* A, float *Z, int hidden_dim);
__global__ void logloss(float* L, float *A, float* y_train, int hidden_dim);
__global__ void rLoss(float *l, float* L);
__global__ void relu(float* A, float* Z, int hidden_dim);

// backward funcitons, please change names afterwards.


__global__ void dA(float* dA, float* A, float* y_true, int hidden_dim);


// Host functions
bool read_mnist_data(
    const std::string& images_path,
    const std::string& labels_path,
    std::vector<float>& images,
    std::vector<float>& labels,
    const int num_images = 0,    // Default to actual MNIST size
    const int image_size = 0     // Default to actual image size
);

void gpuSoftmax(float* data, int batch_size, int hidden_dim);