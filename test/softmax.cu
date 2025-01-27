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


    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDONE) {
        max[0] = Z[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
            max[0] = fmaxf(max[threadIdx.y],Z[row*hidden_dim+i]) ;
        }
        buffPerBlock[0] = 0.0f;
    }
    __syncthreads();

    // maxP
    if (col < hidden_dim && row < BATCH_SIZE) {
        A[row*hidden_dim+col] = exp(Z[row*hidden_dim+col] -  max[threadIdx.y]);
    }
    __syncthreads();
    
    // now Z is useles; I can use it sum over? using a simple reduction scheme?
    // maybe Z is being used eventually.
    // you don't want to keep changing things
    // you stupid fuck
    
    if (col == 0 && row < BATCH_SIZE) {
        buffPerBlock[0] = A[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
            buffPerBlock[0] += A[row*hidden_dim+i];
        }
    }
    __syncthreads();

    // maxP
    if (col < hidden_dim && row < BATCH_SIZE) {
        A[row*hidden_dim+col] /= buffPerBlock[threadIdx.y];
    }
}

// block of 16 x 16
// ok
__global__ void softmaxTiled(float* A, float *Z, int hidden_dim){

    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 

    extern __shared__ float buff[];

	// compute max
    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDIMY) {
        buff[threadIdx.y] = Z[row*hidden_dim];
        float zt = 0.0f;
        for (int i = 1; i < hidden_dim; i++){
            zt = Z[row*hidden_dim+i];
            buff[threadIdx.y] = fmaxf(buff[threadIdx.y],zt);
        }
    }
    __syncthreads();

	// update A: tiled way
    if (col < blockDim.x){ // same shared memory
        for (unsigned int tile = 0 ; tile+col < hidden_dim; tile+=blockDim.x){
			A[row*hidden_dim+col+tile] = exp(Z[row*hidden_dim+col+tile] -  buff[threadIdx.y]);
        }
        //if (col == 0)   buff[threadIdx.y] = 0.0f; // reset for sum
    }
    __syncthreads();

    if (col == 0 && row < BATCH_SIZE && threadIdx.y < BLOCKDIMY) {
		buff[threadIdx.y] = A[row*hidden_dim];
        for (int i = 1; i < hidden_dim; i++){
			buff[threadIdx.y] += A[row*hidden_dim+i];
        }
    }
    __syncthreads();

    if (col < blockDim.x){ 
        for (unsigned int tile = 0 ; tile+col < hidden_dim; tile+=blockDim.x){
			A[row*hidden_dim+col+tile] /= buff[threadIdx.y];
        }
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


bool verifyMaxIndices(const float* A, const float* B, 
                     int batch_size, int output_dim) {
    for (int i = 0; i < batch_size; i++) {
        // Find max indices
        int max_idx1 = 0, max_idx2 = 0;
        float max1 = A[i * output_dim];
        float max2 = B[i * output_dim];
        
        for (int j = 1; j < output_dim; j++) {
            if (A[i * output_dim + j] > max1) {
                max1 = A[i * output_dim + j];
                max_idx1 = j;
            }
            if (B[i * output_dim + j] > max2) {
                max2 = B[i * output_dim + j];
                max_idx2 = j;
            }
        }
        
        if (max_idx1 != max_idx2) {
            printf("Row %d: Different max indices: %d vs %d\n", 
                   i, max_idx1, max_idx2);
            return false;
        }
    }
    return true;
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


void gpuSoftmax(float* data, int batch_size, int hidden_dim) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    // Allocate device memory
    float *d_data;
    size_t matrix_size = batch_size * hidden_dim * sizeof(float);
    cudaMalloc((void**)&d_data, matrix_size);
    cudaMemcpy(d_data, data, matrix_size, cudaMemcpyHostToDevice);

    // Create tensor descriptor for batch_size x hidden_dim matrix
    cudnnTensorDescriptor_t data_desc;
    cudnnCreateTensorDescriptor(&data_desc);
    // NCHW: batch_size x hidden_dim x 1 x 1
    cudnnSetTensor4dDescriptor(
        data_desc,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        batch_size,    // N: number of images
        hidden_dim,    // C: number of channels (features)
        1,            // H: height
        1             // W: width
    );

    // Perform softmax
    float alpha = 1.0f, beta = 0.0f;
    cudnnSoftmaxForward(
        cudnn,
        //CUDNN_SOFTMAX_ACCURATE,
        CUDNN_SOFTMAX_FAST,
        CUDNN_SOFTMAX_MODE_CHANNEL,  // Softmax across hidden_dim
        &alpha,
        data_desc,
        d_data,
        &beta,
        data_desc,
        d_data
    );

    // Copy result back
    cudaMemcpy(data, d_data, matrix_size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_data);
    cudnnDestroyTensorDescriptor(data_desc);
    cudnnDestroy(cudnn);
}



int main(){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    std::vector<float> W(BATCH_SIZE*OUTPUT_DIM);   // 64 x 32
    std::vector<float> b(OUTPUT_DIM);
    utils::xavier_init(W.data(), b.data(),BATCH_SIZE, OUTPUT_DIM);

    std::vector<float> W_cudnn = W; // atomic

    std::vector<float> W_cpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);

    float ms;
    cudaEventRecord(start);

    gpuSoftmax(W_cudnn.data(), BATCH_SIZE, OUTPUT_DIM);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::left << std::setw(30) <<">> cuDNN implementation took" 
                                            << utils::formatTime(ms) << std::endl;




    // CPU time.
    utils::Timer timeCPU = utils::Timer(">> CPU - ");
    cpusmx(W_cpu.data(), W.data(), OUTPUT_DIM);
    std::string cpuTime = timeCPU.report();


    std::cout << "CPU and cudnn implementation difference: "<< returnMaxDiff(W_cpu.data(),W_cudnn.data(), BATCH_SIZE, OUTPUT_DIM, 0)<< "\n";
    float epsilon = 1e-4f;  // Adjust based on precision needs

    // Check each row
    for (int i = 0; i < BATCH_SIZE; i++) {
        // Find max values for each implementation
        float max1 = W_cpu[i * OUTPUT_DIM];
        float max2 = W_cudnn[i * OUTPUT_DIM];
        
        for (int j = 1; j < OUTPUT_DIM; j++) {
            max1 = fmaxf(max1, W_cpu[i * OUTPUT_DIM + j]);
            max2 = fmaxf(max2, W_cudnn[i * OUTPUT_DIM + j]);
        }
        
        // Compare max values
        float max_diff = fabsf(max1 - max2);
        if (max_diff > epsilon) {
            printf("Row %d: Max difference too large: %e\n", i, max_diff);
        }

        // Verify row sums to 1
        float sum1 = 0.0f, sum2 = 0.0f;
        float max_element_diff = 0.0f;
        
        for (int j = 0; j < OUTPUT_DIM; j++) {
            sum1 += W_cpu[i * OUTPUT_DIM + j];
            sum2 += W_cudnn[i * OUTPUT_DIM + j];
            max_element_diff = fmaxf(max_element_diff, 
                                    fabsf(W_cpu[i * OUTPUT_DIM + j] - W_cudnn[i * OUTPUT_DIM + j]));
        }
        
        if (fabsf(sum1 - 1.0f) > epsilon || fabsf(sum2 - 1.0f) > epsilon) {
            printf("Row %d: Sum not close to 1.0: %f, %f\n", i, sum1, sum2);
        }
        
        if (max_element_diff > epsilon) {
            printf("Row %d: Max element difference: %e\n", i, max_element_diff);
        }
    }

    std::cout << "Reality " << verifyMaxIndices(W_cpu.data(), W_cudnn.data(), BATCH_SIZE, OUTPUT_DIM);

    // it's just GPU now.

    std::vector<float> W_gpu(BATCH_SIZE*OUTPUT_DIM, 0.0f);
    std::vector<float> W_gpu2(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f);
    std::vector<float> W_gpu3(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f);
    std::vector<float> W_gpu4(BATCH_SIZE*(OUTPUT_DIM+2), 0.0f); // atomic

    std::vector<float> W_gpu5(BATCH_SIZE*(OUTPUT_DIM), 0.0f); // atomic

    float btoMB = 1024.0f*1024.0f;
    std::cout << "\nMem size for matrix: "  << BATCH_SIZE*OUTPUT_DIM*sizeof(float)/btoMB << " MB\n";
    std::cout << "Mem size for matrix w/ buff: "  << BATCH_SIZE*(OUTPUT_DIM+2)*sizeof(float)/btoMB << "MB\n";

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



    float out = 64.0f;
    float batch = 4.0f;

    dim3 blockDim16(out,batch);     
    dim3 gridDimOB(ceil(OUTPUT_DIM/out),ceil(BATCH_SIZE/batch)); // 2 x 4
    
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

    softmaxTiled<<<gridDimOB, blockDim16, batch*sizeof(float)>>>(A_d, W_d, OUTPUT_DIM); 
    // 285 ms > normal

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
