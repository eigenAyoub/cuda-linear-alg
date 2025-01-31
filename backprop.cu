#include <iostream>
#include "backprop.cuh"
#include <cmath>
#include <vector>
#include <fstream>
#include <cudnn.h>
#include "cuda_runtime.h"
//#include <math_constants.h>

#define TILE_WIDTH 16
#define BATCH_SIZE 64
#define FULL_MASK 0xffffffff
#define WARPS_PER_ROW 1 

// backprop stuff
#define ALPHA 0.0001
#define BETA1 0.9
#define BETA2 0.999
#define EPS   1e-9

// adam working like a charm.
// putting EPS at last (not the canonical form) gives much better resutls.
// not so sure why.

__global__ void 
update1DAdam(float* W, float* dW, float *m, float *v, int step, int x) {
    int row = threadIdx.x + blockDim.x*blockIdx.x;
    if (row<x){
        m[row] = BETA1*m[row] + (1-BETA1)*dW[row];
        v[row] = BETA2*v[row] + (1-BETA2)*dW[row]*dW[row];
        //float mth = m[row]/(1-powf(BETA2, step));
        //float vth = v[row]/(1-powf(BETA2, step));
        //W[row] -= ALPHA*(mth/(sqrtf(vth)+EPS));
        W[row] -= ALPHA*(m[row]/(sqrtf(v[row])+EPS))*(sqrtf(1-powf(BETA2, step))/(1-powf(BETA1, step)));
    }
}

__global__ void 
update2DAdam(float* W, float* dW, float *m, float *v, int step,  int Wy, int Wx) {

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    if (row<Wy && col<Wx){
        m[row*Wx+col] = BETA1*m[row*Wx+col] + (1-BETA1)*dW[row*Wx+col];
        v[row*Wx+col] = BETA2*v[row*Wx+col] + (1-BETA2)*dW[row*Wx+col]*dW[row*Wx+col];
        //float mth = m[row*Wx+col]/(1-powf(BETA2, step));
        //float vth = v[row*Wx+col]/(1-powf(BETA2, step));
        //W[row*Wx + col] -= ALPHA*(mth/(sqrtf(vth)+EPS));

        float alpha_t = ALPHA*(m[row*Wx+col]/(sqrtf(v[row*Wx+col])+EPS));
        W[row*Wx + col] -= alpha_t*(sqrtf(1-powf(BETA2, step))/(1-powf(BETA1, step)));
    }
}

__global__ void 
update1D(float* W, float* dW, int x) {
    int row = threadIdx.x + blockDim.x*blockIdx.x;
    if (row<x){
        W[row] -= 0.0001 * dW[row]; 
    }
}

__global__ void 
update2D(float* W, float* dW, int Wy, int Wx) {

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    if (row<Wy && col<Wx){
        W[row*Wx + col] -= 0.0001 * dW[row*Wx + col];
    }
}


__global__ void
dZ(float* dZ, float* A, float* y_true, int hidden_dim){

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float update = A[row*hidden_dim+col];

    if (row < BATCH_SIZE && col < hidden_dim){
        if (col == static_cast<int>(y_true[row])){
            update -= 1.f;
        } 
        dZ[row*hidden_dim+col] = update / BATCH_SIZE;
    }

}

__global__ void
db(float* db, float* dZ, int hidden_dim){
    // use warm primities here:
    // make hidden dim higher and compare.
    // I only have 10 outputs.
    int row =  threadIdx.x + blockDim.x*blockIdx.x;  // row over Hidden dim 
    if (row < hidden_dim){
        float interVal  = 0.0f;
        for (unsigned int i=0; i < BATCH_SIZE; i++ ){
            interVal += dZ[i*hidden_dim+row];
        }
        db[row] = interVal;
    }
}
// we would like the TILE_WIDTH to be the same as the block width.

__global__ void argmax(float* A, float *Z, int hidden_dim, int warpsPerRow, float *y_true, float *pred){

    int col = threadIdx.x; 
    int row = blockIdx.x; 

    __shared__ float vals[WARPS_PER_ROW];  
    __shared__ float inds[WARPS_PER_ROW];  

    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < hidden_dim);

    float val = -INFINITY;
    int ind   = -1;
    if (col < hidden_dim){
        ind = col;
        val = Z[row*hidden_dim+col];
    }

    if (col < hidden_dim){
        for (unsigned int l = 16; l > 0; l >>= 1){ // you still do over your warp.
            float tempVal = __shfl_down_sync(mask, val, l);
            float tempInd = __shfl_down_sync(mask, ind, l);
            if (tempVal > val){
                val = tempVal;
                ind = tempInd;
            }

        }
    }
    __syncthreads();

    if (col < hidden_dim && threadIdx.x % 32 == 0){
        vals[threadIdx.x/32] = val;  
        //inds[threadIdx.x/32] = (float)ind;  
        inds[threadIdx.x/32] = (float) ind;  
    }
    __syncthreads();

    if (col < warpsPerRow){
        #pragma unroll
        for (int ss = warpsPerRow/2; ss > 0 ; ss >>= 1){
            if (col < ss){
                if (vals[col] < vals[col+ss]){
                    vals[col] = vals[ss+col];
                    inds[col] = inds[ss+col];
                }
            }
        }
    }
    __syncthreads();

    if (col==0){
        //pred[row] = (inds[0]==static_cast<int>(y_true[row]))? 1.0f:0.0f;
        pred[row] = (inds[0]==static_cast<int>(y_true[row]))? 1.0f:0.0f;
    }
}


__global__ void softmax(float* A, float *Z, int hidden_dim, int warpsPerRow){

    int col = threadIdx.x; 
    int row = blockIdx.x; 

    extern __shared__ float red[];  

    unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < hidden_dim);
    float val = Z[row*hidden_dim+col];

    if (col < hidden_dim){
        #pragma unroll
        for (unsigned int l = 16; l > 0; l >>= 1){ // you still do over your warp.
            val = fmaxf(val,__shfl_down_sync(mask, val, l));
        }
    }
    __syncthreads();
    // now each val of the first thread in the warps contains the max per warp.

    if (col < hidden_dim && threadIdx.x % 32 == 0){
        red[threadIdx.x/32] = val;  // red is a 32 size in shared mem; it contains maxes of every entry 
    }
    __syncthreads();

    if (col < warpsPerRow){
        #pragma unroll
        for (int ss = warpsPerRow/2; ss > 0 ; ss >>= 1){
            if (col < ss){
                red[col] = fmaxf(red[col], red[ss+col]);
            }
        }
    }
    __syncthreads();

    if (col < hidden_dim){ // same shared memory
        val = exp(Z[row*hidden_dim+col] - red[0]);
        A[row*hidden_dim+col] = val;
    }
    __syncthreads();


    if (col < hidden_dim){
        #pragma unroll
        for (unsigned int l = 16; l > 0; l >>= 1){
            val += __shfl_down_sync(mask, val, l);
        }
    }
    __syncthreads();

    if (col % 32 == 0 && col < hidden_dim){
        red[threadIdx.x/32] =  val;
    }
    __syncthreads();

    if (col < warpsPerRow){
        #pragma unroll
        for (int ss = warpsPerRow/2; ss > 0 ; ss >>= 1){
            if (col < ss){
                red[col] += red[ss+col];
            }
        }
    }
    __syncthreads();

    if (col < hidden_dim){ 
        A[row*hidden_dim+col] /= red[0];
    }
}


__global__ void
mult_A_T_B(float* A, float* B, float* C, int Ay, int cWidth, int Bx){ // cWidth as common width.

    // multiply C = A.T @ B    (X^T @ dZ)    ([INPUT_DIM, BATCH_SIZE]@[BATCH_SIZE, HIDDEN_DIM])
    // A stored in row-major order.
    // so cWidth should be BATCH_SIZE
    //    Ay               INPUT_DIM

    // A is (Ay, Ax) (in our case, X.T @ dZ)

    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        //Tile_A[tIdy][tIdx] = (row < Ay && tIdx+i < cWidth) ? A[row*cWidth + tIdx + i] : 0.0f;
        if (row < Ay && tIdx+i < cWidth){
            sTile_A[tIdy][tIdx] = A[(tIdx + i)*Ay+ row];
        } else {
            sTile_A[tIdy][tIdx] = 0.0f;
        }
        if (col < Bx && tIdy+i < cWidth){
            sTile_B[tIdy][tIdx] = B[(tIdy+ i)*Bx + col];
        } else {
            sTile_B[tIdy][tIdx] = 0.0f;
        }

        __syncthreads();

        if (col < Bx){
            for (int k=0; k<TILE_WIDTH; ++k){
                interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
            }
        }
        __syncthreads();
    }

    if (row < Ay  && col < Bx){
        C[row*Bx + col] = interVal;
    }
}


// next kernel should have as many threads as the BATCH_SIZE // and just 1D
__global__ void
logloss(float* L, float *A, float* y_train, int hidden_dim){
    // A is assumed to be [BATCH_SIZE x N_CLASSES] // assumed to be the softmax.
    // L = [-log(loss)] // [BATCH_SIZE]

    int row = threadIdx.x + blockDim.x*blockIdx.x;
    if (row<BATCH_SIZE){
        float pred = fmaxf(A[row*hidden_dim+(int)y_train[row]], 1e-30f);
        L[row] = -__logf(pred);
    }
}

__global__ void
shared_bias(float* Z, float* Y, float* b, int hidden_dim){

    int row = blockDim.y*blockIdx.y + threadIdx.y; 
    int col = blockDim.x*blockIdx.x + threadIdx.x; 

    extern __shared__ float bias[];

    int tid = threadIdx.x + blockDim.x*threadIdx.y;

    if (tid < blockDim.x) { bias[threadIdx.x] = b[col]; }
    __syncthreads();

    if (row < BATCH_SIZE && col < hidden_dim){
        Z[row*hidden_dim + col] = Y[row*hidden_dim + col] + bias[threadIdx.x];
    }
    __syncthreads();

}

__global__ void
coalesced_bias(float* Z, float* Y, float* b, int hidden_dim)
{
    int row = blockDim.y * blockIdx.y + threadIdx.y; 
    int col = blockDim.x * blockIdx.x + threadIdx.x; 
    Z[row * hidden_dim + col] = Y[row * hidden_dim + col] + b[col];
}

__global__ void
mult(float* A, float* B, float* C, int Ay, int cWidth, int Bx){ // cWidth as common width.

    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < Ay && tIdx+i < cWidth) ? A[row*cWidth + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < Bx && tIdy+i < cWidth) ? B[(tIdy+ i)*Bx + col] : 0.0f;
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
        }
        __syncthreads();
    }

    if (row < Ay  && col < Bx){
        C[row*Bx + col] = interVal;
    }
}

__global__ void
mult_A_B_T(float* A, float* B, float* C, int Ay, int cWidth, int Bx){ // cWidth as common width.
    // B^T is of shape cWidth x Bx

    __shared__ float sTile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sTile_B[TILE_WIDTH][TILE_WIDTH];

    int tIdy = threadIdx.y; 
    int tIdx  = threadIdx.x;

    int row = threadIdx.y + blockDim.y*blockIdx.y;
    int col = threadIdx.x + blockDim.x*blockIdx.x;

    float interVal = 0 ;

    for (int i= 0; i < cWidth; i+= TILE_WIDTH){
        sTile_A[tIdy][tIdx] = (row < Ay && tIdx+i < cWidth) ? A[row*cWidth + tIdx + i] : 0.0f;
        sTile_B[tIdy][tIdx] = (col < Bx && tIdy+i < cWidth) ? B[(tIdy+ i)+ col*cWidth] : 0.0f;
        //B^T[(tIdy+ i)][col] : 0.0f;
        // = B[col][(tIdy+ i)] : 0.0f;
        // = B[col*cWidth + tIdy + i]
        __syncthreads();

        for (int k=0; k<TILE_WIDTH; ++k){
            interVal += sTile_A[tIdy][k]*sTile_B[k][tIdx];
        }
        __syncthreads();
    }

    if (row < Ay  && col < Bx){
        C[row*Bx + col] = interVal;
    }
}

__global__ void
rLoss(float *l, float* L){

    int row = threadIdx.x + blockDim.x*blockIdx.x;

    for (int i = BATCH_SIZE/2; i > 0; i = i/2){
        if (row < i) {
            L[row] += L[row+i];
        }
    }
    __syncthreads();

    if (row==0){
        l[0] =  L[0]/BATCH_SIZE;
    }

}

__global__ void
relu(float* A, float* Z, int hidden_dim, int B){

    int row = blockIdx.y*blockDim.y + threadIdx.y ;
    int col = blockIdx.x*blockDim.x + threadIdx.x ;

    if (row < B && col < hidden_dim){
        A[row*hidden_dim+col]  = (Z[row*hidden_dim+col] > 0)? Z[row*hidden_dim+col]:0.0f;
    }
}

__global__ void
dRelu(float *dA, float *Z, float *dZ, int hidden_dim){

    int row = blockIdx.y*blockDim.y + threadIdx.y ;
    int col = blockIdx.x*blockDim.x + threadIdx.x ;

    if (row < BATCH_SIZE && col < hidden_dim){
        dZ[row*hidden_dim+col]  = (Z[row*hidden_dim+col] > 0)? dA[row*hidden_dim+col]:0.0f;
    }
}

bool read_mnist_data(
    const std::string& images_path,
    const std::string& labels_path,
    std::vector<float>& images,
    std::vector<float>& labels,
    const int num_images,
    const int image_size   
) {
    // Open files
    std::cout << num_images << image_size << "\n";

    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file || !labels_file) {
        std::cerr << "Error opening MNIST files" << std::endl;
        return false;
    }

    // Create temporary buffers
    std::vector<uint8_t> images_buff(num_images * image_size);
    std::vector<uint8_t> labels_buff(num_images);

    // Read binary data
    images_file.read(reinterpret_cast<char*>(images_buff.data()), 
                    num_images * image_size);
    labels_file.read(reinterpret_cast<char*>(labels_buff.data()), 
                    num_images);

    // Resize output vectors
    images.resize(num_images * image_size);
    labels.resize(num_images);

    // Convert to float
    std::copy(images_buff.begin(), images_buff.end(), images.begin());
    std::copy(labels_buff.begin(), labels_buff.end(), labels.begin());

    return true;
}


void gpuSoftmax(float* data, int batch_size, int hidden_dim) {
    // thanks Claude
    // Create cuDNN handle
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
        CUDNN_SOFTMAX_LOG,
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
