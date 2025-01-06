
#include "backprop.cuh"
#include <cmath>

#include "cuda_runtime.h"

#define TILE_WIDTH 16 
#define BATCH_SIZE 32

// borowing notation freely from Izzat El Hajj.
// https://www.youtube.com/@ielhajj
__global__ void
multiply(float* A, float* B, float* C, int image_width){

    __shared__ float TILE_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float TILE_B[TILE_WIDTH][TILE_WIDTH];

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    int tIdy = threadIdx.y;
    int tIdx  = threadIdx.x;

    int bIdy  = blockIdx.y;
    int bIdx  = blockIdx.x;


    // for () how many passes do we need until we are done with one TILE going thorugh all dim
    int TILE_SIZE = 16;
    int intC = 0;
    int steps = 4; // width   A = [x, 16*4] // B = [16x4,x]
    for (int step = 0; step < 4; step+=TILE_SIZE){

        TILE_A[threadIdx.x][threadIdx.y] = A[1];
        TILE_B[threadIdx.x][threadIdx.y] = B[1];

        __syncthreads();

        for (int k=0; k < TILE_WIDTH; ++k){
            intC += TILE_A[tIdy][k]*TILE_B[k][tIdx];
        }
        __syncthreads();

        C[row*image_width+col] = intC;
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
