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

const int BATCH_SIZE = 2048;
const int OUTPUT_DIM = 1024;

const int BLOCKDIMY  = 16;
const int BLOCKDONE  = 1;


__global__ void softmaxShared(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    __shared__ float buffPerBlock[1];
    __shared__ float max[1];


    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDIMY) {
        max[0] = Z[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
            float zt = Z[row*hidden_dim+i];
            max[0] = fmaxf(max[threadIdx.y],zt) ;
        }
        buffPerBlock[0] = 0.0f;
    }
    __syncthreads();

    // maxP
    if (col < hidden_dim && row < BATCH_SIZE) {
        A[row*hidden_dim+col] = exp(Z[row*hidden_dim+col] -  max[threadIdx.y]);
    }
    __syncthreads();

    if (col < hidden_dim && row < BATCH_SIZE) {
        atomicAdd(&buffPerBlock[threadIdx.y], A[row*hidden_dim+col]);
    }
    __syncthreads();

    // maxP
    if (col < hidden_dim && row < BATCH_SIZE) {
        A[row*hidden_dim+col] /= buffPerBlock[threadIdx.y];
    }
}

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


__global__ void softmaxWithDRAMBuffer(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDONE) {
        A[(row+1)*(hidden_dim)-1] = Z[row*(hidden_dim-2)];
        for (int i = 1; i < (hidden_dim-2); i++){
            A[(row+1)*(hidden_dim)-1] = fmaxf(A[(row+1)*(hidden_dim)-1], Z[row*(hidden_dim-2)+i]) ;
        }
        A[(row+1)*(hidden_dim)-2] = 0.0f; // sum buffer.
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < (hidden_dim-2); i++){
            float At = exp(Z[row*(hidden_dim-2)+i] - A[(row+1)*hidden_dim-1]); // At = exp(Z - maxPerRow)
            A[row*hidden_dim+i] = At;
            sum += At;
        }  
        A[(row+1)*(hidden_dim)-2] = sum;
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        for (int i = 0; i < (hidden_dim-2); i++){
            A[row*hidden_dim+i] = A[row*hidden_dim+i]/A[(row+1)*(hidden_dim)-2];
        }
    }
}

__global__ void softmaxWithDRAMBuffer2(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDONE) {
        A[(row+1)*(hidden_dim)-1] = Z[row*(hidden_dim-2)];
        for (int i = 1; i < (hidden_dim-2); i++){
            A[(row+1)*(hidden_dim)-1] = fmaxf(A[(row+1)*(hidden_dim)-1], Z[row*(hidden_dim-2)+i]) ;
        }
        A[(row+1)*(hidden_dim)-2] = 0.0f; // sum buffer.
    }
    __syncthreads();

    if (row < BATCH_SIZE && col < hidden_dim-2) {
        A[row*hidden_dim+col] = exp(Z[row*(hidden_dim-2)+col] - A[(row+1)*hidden_dim-1]); // At = exp(Z - maxPerRow)
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < (hidden_dim-2); i++){
            sum += A[row*hidden_dim+i] ;
        }  
        A[(row+1)*(hidden_dim)-2] = sum;
    }
    __syncthreads();


    if (row < BATCH_SIZE && col < hidden_dim-2) {
        A[row*hidden_dim+col] /= A[(row+1)*hidden_dim-2]; // At /= maxPerRow 
    }
}


__global__ void softmaxWithDRAMBufferAtomic(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDONE) {
        float maxVal = Z[row*(hidden_dim-2)];
        for (int i = 1; i < (hidden_dim-2); i++){
            maxVal = fmaxf(maxVal, Z[row*(hidden_dim-2)+i]) ;
        }
        A[(row+1)*(hidden_dim)-1] = maxVal;
        //A[(row+1)*(hidden_dim)-2] = 0.0f; 
    }
    __syncthreads();

    if (row < BATCH_SIZE && col < hidden_dim-2) {
        float v = exp(Z[row*(hidden_dim-2)+col] - A[(row+1)*hidden_dim-1]); // At = exp(Z - maxPerRow)
        A[row*hidden_dim+col] = v ;
        atomicAdd(&A[(row+1)*hidden_dim - 2], v);
    }
    __syncthreads();

    if (row < BATCH_SIZE && col < hidden_dim-2) {
        A[row*hidden_dim+col] /= A[(row+1)*hidden_dim-2]; // At /= maxPerRow 
    }
}

float returnMaxDiff(float * A, float * B, int bs, int dA, int d){

    // bs: batch size :p
    // A gpu matrix
    // dA width of A
    // d = dA - dB

    float diff = 1e-30f;
    float tempDiff = 0.0f;

    for (unsigned int r  = 0; r < bs; r++){
        for (unsigned int c = 0; c  < dA-d; c++){
            tempDiff = std::abs(A[r*dA+c]-B[r*(dA-d)+c]);
            if (tempDiff > diff)    diff = tempDiff;
        }
    }
    return diff;
}

int main(){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> W(BATCH_SIZE*OUTPUT_DIM);   // 64 x 32
    std::vector<float> b(OUTPUT_DIM);
    utils::xavier_init(W.data(), b.data(),BATCH_SIZE, OUTPUT_DIM);


    std::vector<float> W_cpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);


    // CPU time.
    utils::Timer timeCPU = utils::Timer(">> CPU - ");
    cpusmx(W_cpu.data(), W.data(), OUTPUT_DIM);
    std::string cpuTime = timeCPU.report();



    // it's just GPU now.

    std::vector<float> W_gpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);
    std::vector<float> W_gpu2(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f);
    std::vector<float> W_gpu3(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f);
    std::vector<float> W_gpu4(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f); // atomic

    std::vector<float> W_gpu5(BATCH_SIZE*(OUTPUT_DIM), 0.0f); // atomic

    float btoMB = 1024.0f*1024.0f;
    std::cout << "\nMem size for matrix: "  << BATCH_SIZE*OUTPUT_DIM*sizeof(float)/btoMB << " MB\n";
    std::cout << "Mem size for matrix w/ buff: "  << BATCH_SIZE*(OUTPUT_DIM+2)*sizeof(float)/btoMB << "MB\n";

    float ms;
    float *W_d;
    float *A_d;  // we bringing this back.
    float *A_d2; // we bringing this back.
    float *A_d3; // we bringing this back.
    float *A_d4; // we bringing this back.
    float *A_d5; // we bringing this back.

    cudaMalloc((void **) &W_d,  sizeof(float)*W.size());
    cudaMalloc((void **) &A_d,  sizeof(float)*W.size());
    cudaMalloc((void **) &A_d2, sizeof(float)*W_gpu2.size());  // [BATCH_SIZE x OUTPUT_DIM+2]
    cudaMalloc((void **) &A_d3, sizeof(float)*W_gpu3.size());  // [BATCH_SIZE x OUTPUT_DIM+2]
    cudaMalloc((void **) &A_d4, sizeof(float)*W_gpu4.size());  // [BATCH_SIZE x OUTPUT_DIM+2]
    cudaMalloc((void **) &A_d5, sizeof(float)*W_gpu5.size());  // [BATCH_SIZE x OUTPUT_DIM+2]

    cudaMemset(A_d4, 0,sizeof(float)*W_gpu4.size());


    cudaEventRecord(start);           

    cudaMemcpy(W_d, W.data(), sizeof(float)*W.size(), cudaMemcpyHostToDevice);

    cudaEventRecord(stop);           
    cudaEventSynchronize(stop);      
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) << "Time to cudaMemcpy: " 
                                            << utils::formatTime(ms) << std::endl;
    std::cout << "\n";



    dim3 blockDim16(16,16);     
    dim3 gridDimOB(ceil(OUTPUT_DIM/16.0f),ceil(BATCH_SIZE/16.0f)); // 2 x 4

    // warm-up.
    for (int i = 0; i < 20; i++){
        softmax<<<gridDimOB, blockDim16>>>(A_d, W_d, OUTPUT_DIM); 
    }
    cudaDeviceSynchronize();

    // cpu time:
    std::cout << "Time of different implementations:\n\n";
    std::cout << timeCPU.report()  << std::endl;

    // GPU - CUDA, Naive
    cudaEventRecord(start);

    softmax<<<gridDimOB, blockDim16>>>(A_d, W_d, OUTPUT_DIM); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) <<">> GPU - naive softmax: " 
                                            << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu.data(), A_d, sizeof(float)*W.size(), cudaMemcpyDeviceToHost);


    // GPU - CUDA, shared; one block per row 

    cudaEventRecord(start);
    dim3 blockDimO(OUTPUT_DIM,1);     
    dim3 gridDimO(1, BATCH_SIZE); 

    softmaxShared<<<gridDimO, blockDimO>>>(A_d5, W_d, OUTPUT_DIM); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) <<">> GPU - softmax sharedmem: " 
                                            << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu5.data(), A_d5, sizeof(float)*W.size(), cudaMemcpyDeviceToHost);


    
    // GPU - CUDA, Naive, with buffer in DRAM, and no shared memory usage.
    dim3 blockDim11(OUTPUT_DIM,1);     
    dim3 gridDim11(1,BATCH_SIZE);     

    cudaEventRecord(start);

    softmaxWithDRAMBuffer<<<gridDim11, blockDim11>>>(A_d2, W_d, OUTPUT_DIM+2); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::left << std::setw(30) 
                            << ">> GPU - softmax with buffer: " 
                            << utils::formatTime(ms) << std::endl;

    cudaMemcpy(W_gpu2.data(), A_d2, sizeof(float)*W_gpu2.size(), cudaMemcpyDeviceToHost);


    // GPU - CUDA, Naive, with buffer in DRAM, and no shared memory usage // slightly more intel.
    cudaEventRecord(start);

    softmaxWithDRAMBuffer2<<<gridDim11, blockDim11>>>(A_d3, W_d, OUTPUT_DIM+2); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::left << std::setw(30) << ">> GPU - softmax with buffer+ " 
                                            << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu3.data(), A_d3, sizeof(float)*W_gpu3.size(), cudaMemcpyDeviceToHost);

    // GPU - Atomic DRAM buffer 
    cudaEventRecord(start);

    softmaxWithDRAMBufferAtomic<<<gridDim11, blockDim11>>>(A_d4, W_d, OUTPUT_DIM+2); 

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << std::left << std::setw(30) << ">> GPU - softmax atomic buffer+ " 
                                            << utils::formatTime(ms) << std::endl;
    cudaMemcpy(W_gpu4.data(), A_d4, sizeof(float)*W_gpu4.size(), cudaMemcpyDeviceToHost);


    std::cout << "\nMax of absolute distance:\n";
    /// Computing the diffs per implementation.

    float diff = 0.0f;

    diff = returnMaxDiff(W_gpu.data(), W_cpu.data(), BATCH_SIZE, OUTPUT_DIM, 0);
    std::cout << "> max(abs(CPU - GPU_NAIVE)) = " << diff << "\n";

    diff = returnMaxDiff(W_gpu5.data(), W_cpu.data(), BATCH_SIZE, OUTPUT_DIM, 0);
    std::cout << "> max(abs(CPU - GPU_shared) = " << diff << "\n";

    diff = returnMaxDiff(W_gpu2.data(), W_cpu.data(), BATCH_SIZE, OUTPUT_DIM+2, 2);
    std::cout << "> max(abs(CPU - GPU_Buff) = " << diff << "\n";

    diff = returnMaxDiff(W_gpu3.data(), W_cpu.data(), BATCH_SIZE, OUTPUT_DIM+2, 2);
    std::cout << "> max(abs(CPU - GPU_Buff_plus) = " << diff << "\n";

    diff = returnMaxDiff(W_gpu4.data(), W_cpu.data(), BATCH_SIZE, OUTPUT_DIM+2, 2);
    std::cout << "> max(abs(CPU - GPU_atomic) = " << diff << "\n";

}