
#include "backprop.cuh"
#include <cmath>

#include "cuda_runtime.h"

#define TILE_WIDTH 32  

// we would like the TILE_WIDTH to be the same as the block width.
// so far we assume that the matrix is squared N x N

#define BATCH_SIZE 32

// Borowing ideas from Izzat El Hajj.
// https://www.youtube.com/@ielhajj

__global__ void
mult(float* A, float* B, float* C, int N){
    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i< N; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < N && tIdx+i < N) ? A[row*N + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < N && tIdy+i < N) ? B[(tIdy+ i)*N + col] : 0.0f;
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
        }
        __syncthreads();
    }

    if (row<N && col < N){
        C[row*N + col] = interVal;
    }
}

__global__ void
relu(float* A, float* Z, int B, int F){
    // takes pre-activations Z
    // computes the activation function  A = relu(Z)
    // B: batch size
    // F: Feature size 
    int tIdx = blockIdx.x*blockDim.x + threadIdx.x ;
    int tIdy = blockIdx.y*blockDim.y + threadIdx.y ;
    if (tIdx< B && tIdy < F){
        A[tIdx] = (Z[tIdx]>0) ? Z[tIdx] : 0;
    }
    
}

__global__ void
softmax(float* I, int *S, int width){
    // should compute the logits 
}

__global__ void
rSoftmax(float* S, int *Y, float* rS, int B){
    // takes softmax logits S ~ [B,10]
    // returns rS ~ [B] > for each sample S_i,  it select pred_i = S_i[true_y] 
    // and return -log(pred_i)
    int indx = blockIdx.x*blockDim.x + threadIdx.x ;

    if (indx < BATCH_SIZE){
        rS[indx] = -logf(S[Y[indx]]);
    }
}

__global__ void
reduced_sum(float* rS, float* loss, int B){
    // implement the reduction loss pattern
}
