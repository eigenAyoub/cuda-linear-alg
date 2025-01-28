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
__global__ void softmax(float* A, float *Z, int hidden_dim, int warpsPerRow);
__global__ void argmax(float* A, float *Z, int hidden_dim, int warpsPerRow, float * y_true, float *pred);
__global__ void logloss(float* L, float *A, float* y_train, int hidden_dim);
__global__ void rLoss(float *l, float* L);
__global__ void relu(float* A, float* Z, int hidden_dim);

// backward funcitons, please change names afterwards.

__global__ void dZ(float* dZ, float* A, float* y_true, int hidden_dim);
__global__ void mult_A_B_T(float* A, float* B, float* C, int Ay, int cWidth, int Bx); // Bx should be Be rather B_Tx = By
__global__ void mult_A_T_B(float* A, float* B, float* C, int Ay, int cWidth, int Bx); // cWidth as common width.
__global__ void dRelu(float *dA, float *Z, float *dZ, int hidden_dim);

__global__ void db(float* db, float* dZ, int hidden_dim);

// upadtes:
__global__ void update1D(float* W, float* dW, int x);
__global__ void update2D(float* W, float* dW, int y, int x);


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