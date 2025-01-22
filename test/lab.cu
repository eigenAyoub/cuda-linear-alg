#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cudnn.h>
#include <unistd.h> 
#include <iomanip>  

#include "../utils.hpp"

const int BATCH_SIZE = 465;
const int OUTPUT_DIM = 345 ;
const int BLOCKDIMY  = 16;

__global__ void softmax(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    __shared__ float buffPerBlock[BLOCKDIMY];
    __shared__ float max[BLOCKDIMY];

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDIMY) {
        max[threadIdx.y] = Z[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
            float zt = Z[row*hidden_dim+i];
            max[threadIdx.y] = fmaxf(max[threadIdx.y],zt) ;
        }
        buffPerBlock[threadIdx.y] = 0.0f;
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++){
            float At = exp(Z[row*hidden_dim+i] -  max[threadIdx.y]);
            A[row*hidden_dim+i] = At;
            sum += At;
        }
        buffPerBlock[threadIdx.y] = sum;
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        for (int i = 0; i < hidden_dim; i++){
            A[row*hidden_dim+i] = A[row*hidden_dim+i]/buffPerBlock[threadIdx.y];
        }
    }
}

__global__ void atomicSoftmaxV2(float* A, float *Z, int hidden_dim){

    // Z original: [BATCH_SIZE, OUTPUT_DIM]
    // A softmax:  [BATCH_SIZE, OUTPUT_DIM+2]
    // I use the last        col to store the maximum value per row.
    // .  .   .  before last col to store the sum per row of the values.

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (col < hidden_dim-2){
        atomicAdd(&A[row*hidden_dim-2], A[row*hidden_dim+col]);
        atomicAdd(&A[row*hidden_dim-1], A[row*hidden_dim+col]);
    }
  

    // exponential;
    if (row < BATCH_SIZE && col < hidden_dim - 2) {
        A[row*hidden_dim+col] = expf(Z[row*(hidden_dim-2)+col] - A[(row+1)*hidden_dim-1]);
    }
    __syncthreads();

    // computing the sum >  still dumb af
    if (col == 0 && row < BATCH_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim-2; i++){
            sum += A[row*hidden_dim+i];
        }
        A[(row+1)*hidden_dim-2] = sum;
    }
    __syncthreads();

    // exponential;
    if (row < BATCH_SIZE && col < hidden_dim - 2) {
        A[row*hidden_dim+col] /= A[(row+1)*hidden_dim-2];
    }
    __syncthreads();
}


void cpusmx(float *A, float *Z, int outDim, int B = BATCH_SIZE){
    for (int i = 0; i < B; ++i) {
        float maxVal = Z[i * outDim + 0];
        for (int j = 1; j < outDim; ++j) {
            if(Z[i*outDim+j] > maxVal) {
                maxVal = Z[i*outDim+j];
            }
        }
        float sumExp = 0.f;
        for (int j = 0; j < outDim; ++j) {
            float e = std::exp(Z[i * outDim + j] - maxVal);
            A[i * outDim + j] = e;
            sumExp += e;
        }
        for (int j = 0; j < outDim; ++j) {
            A[i * outDim + j] /= sumExp;
        }
    }
}

void back(int d1, int d2, float* dev_var, std::string vName){

    std::vector<float> vBack(d1*d2);
    cudaMemcpy(vBack.data(), dev_var, sizeof(float)*d1*d2, cudaMemcpyDeviceToHost);

    int x  = min(d1, 100);

    std::cout << "\n" << vName << " : \n";
    for (int i=0; i < d1; i++){
        for (int j=0; j < d2; j++){
            std::cout << vBack[i*d2+j] << " ";
        }
        std::cout <<"\n";
    }
    std::cout <<"\n";
}

void visual(int d1, int d2, float* dev_var, std::string vName){

    std::cout << "\n" << vName << " : \n";
    for (int i=0; i < d1; i++){
        for (int j=0; j < d2; j++){
            std::cout << dev_var[i*d2+j] << " ";
        }
        std::cout <<"\n";
    }
    std::cout <<"\n";
}


__global__ void atomicSoftmax(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    __shared__ float buffPerBlock[BLOCKDIMY];
    __shared__ float max[BLOCKDIMY];

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDIMY) {
        max[threadIdx.y] = Z[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
            float zt = Z[row*hidden_dim+i];
            max[threadIdx.y] = fmaxf(max[threadIdx.y],zt) ;
        }
        buffPerBlock[threadIdx.y] = 0.0f;
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < hidden_dim; i++){
            float At = exp(Z[row*hidden_dim+i] -  max[threadIdx.y]);
            A[row*hidden_dim+i] = At;
            sum += At;
        }
        buffPerBlock[threadIdx.y] = sum;
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        for (int i = 0; i < hidden_dim; i++){
            A[row*hidden_dim+i] = A[row*hidden_dim+i]/buffPerBlock[threadIdx.y];
        }
    }
}
int main(){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> W(BATCH_SIZE*OUTPUT_DIM);   // 64 x 32
    std::vector<float> b(OUTPUT_DIM);
    utils::xavier_init(W.data(), b.data(),BATCH_SIZE, OUTPUT_DIM);


    std::vector<float> W_gpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);
    std::vector<float> W_cpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);
    std::vector<float> W_gpu2(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f);

    float btoMB = 1024.0f*1024.0f;
    std::cout << "Mem size for matrix: "  << BATCH_SIZE*OUTPUT_DIM*sizeof(float)/btoMB << " MB \n";
    std::cout << "Mem size for matrix w/ buffer: "  << BATCH_SIZE*(OUTPUT_DIM+2)*sizeof(float)/btoMB << "MB \n";

    float ms;
    float *W_d;
    float *A_d;  // we bringing this back.
    float *A_d2; // we bringing this back.

    cudaMalloc((void **) &W_d,  sizeof(float)*W.size());
    cudaMalloc((void **) &A_d,  sizeof(float)*W.size());
    cudaMalloc((void **) &A_d2, sizeof(float)*W_gpu2.size());  // [BATCH_SIZE x OUTPUT_DIM+2]


    cudaEventRecord(start);           

    cudaMemcpy(W_d, W.data(), sizeof(float)*W.size(), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);           
    cudaEventSynchronize(stop);      
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) << "Time to cudaMemcpy: " 
                                            << utils::formatTime(ms) << std::endl;


    utils::Timer timeCPU = utils::Timer("CPU: ");
    cpusmx(W_cpu.data(), W.data(), OUTPUT_DIM);
    timeCPU.report(); 

    dim3 blockDim16(16,16);     
    dim3 gridDimOB(ceil(OUTPUT_DIM/16.0f),ceil(BATCH_SIZE/16.0f)); // 2 x 4


    for (int i = 0; i < 20; i++){
        softmax<<<gridDimOB, blockDim16>>>(A_d, W_d, OUTPUT_DIM); 
    }
    cudaDeviceSynchronize();




    cudaEventRecord(start);
    softmax<<<gridDimOB, blockDim16>>>(A_d, W_d, OUTPUT_DIM); 
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) <<"GPU - naive softmax: " 
                                            << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu.data(), A_d, sizeof(float)*W.size(), cudaMemcpyDeviceToHost);


    float diff = 1e-30f;
    float tempDiff = 0.0f;

    for (unsigned int r  = 0; r < BATCH_SIZE; r++){
        for (unsigned int c = 0; c  <OUTPUT_DIM; c++){
            tempDiff = std::abs(W_gpu[r*OUTPUT_DIM+c]-W_cpu[r*OUTPUT_DIM+c]);
            if (tempDiff > diff)    diff = tempDiff;
            //std::cout << W_gpu2[r*(OUTPUT_DIM+2)+c] << " " 
            //          << W_cpu[r*OUTPUT_DIM+c]      << "\n";
        }
    }
    std::cout << "> max(abs(CPU - GPU_NAIVE)) = " << diff << "\n";

    
    /** 
     * 
    
    dim3 blockDim11(1024,1);     
    dim3 gridDim11(1,1024);     

    cudaEventRecord(start);

    softmaxV2<<<gridDim11, blockDim11>>>(A_d2, W_d, OUTPUT_DIM+2); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "GPU - half naive softmax: " << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu2.data(), A_d2, sizeof(float)*W_gpu2.size(), cudaMemcpyDeviceToHost);
    for (unsigned int r  = 0; r < BATCH_SIZE; r++){
        for (unsigned int c = 0; c  <OUTPUT_DIM; c++){
            tempDiff = std::abs(W_gpu2[r*(OUTPUT_DIM+2)+c]-W_cpu[r*OUTPUT_DIM+c]);
            if (tempDiff > diff)    diff = tempDiff;
        }
    }
    std::cout << "> max(abs(CPU - GPU_HALF_NAIVE) = " << diff << "\n";
    */
}