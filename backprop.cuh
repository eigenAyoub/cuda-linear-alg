#pragma once

#include <vector>
#include <string>
#include <cuda_runtime.h>

// Constants (if needed globally)
#define TILE_WIDTH 16

// Kernel declarations
__global__ void mult(float* A, float* B, float* C, int Ax, int cWidth, int By);
__global__ void relu(float* A, float* Z, int B, int F);
__global__ void rSoftmax(float* S, float* rS, int B);
__global__ void softmax(float* A, float *Z, float* buffer);

// Host functions
bool read_mnist_data(
    const std::string& images_path,
    const std::string& labels_path,
    std::vector<float>& images,
    std::vector<float>& labels,
    const int num_images = 0,    // Default to actual MNIST size
    const int image_size = 0     // Default to actual image size
);

void gpuSoftmax(float* data, int size);