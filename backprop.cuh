#pragma once

__global__ void multiply(float* A, float* B, float* C);
__global__ void relu(float* A, float* Z, int B, int F);
__global__ void rSoftmax(float* S, float* rS, int B);
