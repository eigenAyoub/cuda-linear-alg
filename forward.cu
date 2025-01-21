#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <cudnn.h>

#include "backprop.cuh"
#include "utils.hpp"

const int IMAGE_SIZE  = 784;
const int NUM_IMAGES  = 60000;

const int INPUT_DIM  = 784;
const int HIDDEN_DIM = 256;
const int OUTPUT_DIM = 10;

const int BATCH_SIZE = 64;

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
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

#define CHECK_KERNEL() { \
    cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("Kernel error %s:%d: %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}
int main(){
    std::vector<float> X_train(NUM_IMAGES*IMAGE_SIZE), y_train(NUM_IMAGES);

    if (!read_mnist_data("data/train_mnist_images.bin",
                         "data/train_mnist_labels.bin",
                          X_train, 
                          y_train,
                          NUM_IMAGES,
                          IMAGE_SIZE
                        )) {
            return -1;
        }

    // first batch //
    std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);  // Batch_size (y)  x INPUT_DIM (x) >> [64, 784jj]
    std::vector<float> y_batch(BATCH_SIZE);              // Batch_size (y)  x INPUT_DIM (x) >> [64, 784]

    std::copy(X_train.begin(), X_train.begin() + BATCH_SIZE * INPUT_DIM, X_batch.begin());
    std::copy(y_train.begin(), y_train.begin() + BATCH_SIZE, y_batch.begin());

    std::vector<float> W1_h(INPUT_DIM*HIDDEN_DIM);
    std::vector<float> b1_h(HIDDEN_DIM);
    utils::xavier_init(W1_h.data(), b1_h.data(), INPUT_DIM, HIDDEN_DIM);

    std::vector<float> W2_h(HIDDEN_DIM*OUTPUT_DIM);
    std::vector<float> b2_h(OUTPUT_DIM);
    utils::xavier_init(W2_h.data(), b2_h.data(),HIDDEN_DIM, OUTPUT_DIM);


    float *X_train_d, *y_train_d;

    //forward stuf, 
    float *W1_d, *b1_d, *Y1_d, *Z1_d, *A1_d;  
    float *W2_d, *b2_d, *Y2_d, *Z2_d, *A2_d;  

    // Y1_h = X @ W1_h   >> [B, 10] >> [64x10]
    // Z1_h = Y1_h + b1_h 
    // A1_h = activation(Z1_h) // Relu then softmax.

    float* L, *l;      
    
    //TODO:
    //  drop the L eventually, go for the l.
    //  fuse X@W+b as one op. 

    cudaMalloc((void **) &X_train_d, sizeof(float)*X_batch.size());
    cudaMalloc((void **) &y_train_d, sizeof(float)*y_batch.size());

    // first layer
    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &A1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    // second layer
    cudaMalloc((void **) &W2_d, sizeof(float)*W2_h.size());
    cudaMalloc((void **) &b2_d, sizeof(float)*b2_h.size());
    cudaMalloc((void **) &Y2_d, sizeof(float)*BATCH_SIZE*OUTPUT_DIM);
    cudaMalloc((void **) &Z2_d, sizeof(float)*BATCH_SIZE*OUTPUT_DIM);
    cudaMalloc((void **) &A2_d, sizeof(float)*BATCH_SIZE*OUTPUT_DIM);

    cudaMalloc((void **) &L, sizeof(float)*BATCH_SIZE);
    cudaMalloc((void **) &l, sizeof(float));

    // copy weights.
    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, b1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(W2_d, W2_h.data(), W2_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b2_d, b2_h.data(), b2_h.size()*sizeof(float), cudaMemcpyHostToDevice);

    //back(INPUT_DIM, HIDDEN_DIM, W1_d);

    // backprop stuf, 
    float *dW1_d, *db1_d, *dZ1_d, *dA1_d;
    float *dW2_d, *db2_d, *dZ2_d;// *dA2_d;

    cudaMalloc((void **) &dW1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &db1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &dZ1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);
    cudaMalloc((void **) &dA1_d, sizeof(float)*BATCH_SIZE*HIDDEN_DIM);

    cudaMalloc((void **) &dW2_d, sizeof(float)*W2_h.size());
    cudaMalloc((void **) &db2_d, sizeof(float)*b2_h.size());
    cudaMalloc((void **) &dZ2_d, sizeof(float)*BATCH_SIZE*OUTPUT_DIM);
    //cudaMalloc((void **) &dA2_d, sizeof(float)*BATCH_SIZE*OUTPUT_DIM);

    dim3 blockDim16(16,16);     

    dim3 gridDimHB(ceil(HIDDEN_DIM/16.0f),ceil(BATCH_SIZE/16.0f)); // 16 x 4
    dim3 gridDimOB(ceil(OUTPUT_DIM/16.0f),ceil(BATCH_SIZE/16.0f)); // 1 x 4
    dim3 gridDimOH(ceil(OUTPUT_DIM/16.0f),ceil(HIDDEN_DIM/16.0f)); //1 x 16
    dim3 gridDimHI(ceil(HIDDEN_DIM/16.0f),ceil(INPUT_DIM/16.0f)); // 16 x 49

    printf("Launch config HB: blocks(%d,%d) threads(%d,%d)\n", 
       gridDimHB.x, gridDimHB.y, blockDim16.x, blockDim16.y);

    printf("Launch config OB: blocks(%d,%d) threads(%d,%d)\n", 
       gridDimOB.x, gridDimOB.y, blockDim16.x, blockDim16.y);
    

    printf("Launch config HI: blocks(%d,%d) threads(%d,%d)\n", 
       gridDimHI.x, gridDimHI.y, blockDim16.x, blockDim16.y);

    for (unsigned int batch = 0 ; batch < 50; batch++){

        CHECK_CUDA(cudaMemcpy(X_train_d, X_train.data()+batch*BATCH_SIZE*INPUT_DIM, X_batch.size()*sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(y_train_d, y_train.data()+batch*BATCH_SIZE, y_batch.size()*sizeof(float), cudaMemcpyHostToDevice));

        // forward pass

        // first layer
        mult<<<gridDimHB, blockDim16>>>(X_train_d, W1_d, Y1_d, 
                        BATCH_SIZE, INPUT_DIM, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        coalesced_bias<<<gridDimHB, blockDim16>>>(Z1_d, Y1_d, b1_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());


        relu<<<gridDimHB, blockDim16>>>(A1_d, Z1_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        // second layer
        mult<<<gridDimOB, blockDim16>>>(A1_d, W2_d, Y2_d, BATCH_SIZE, HIDDEN_DIM, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        //back(2, HIDDEN_DIM, Y2_d, "Y2_d, after mult");

        coalesced_bias<<<gridDimOB, blockDim16>>>(Z2_d, Y2_d, b2_d, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        //back(2, HIDDEN_DIM, Z2_d, "Z2_d, after + b2_d");

        softmax<<<gridDimOB, blockDim16>>>(A2_d, Z2_d, OUTPUT_DIM); 
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        //back(8, OUTPUT_DIM, A2_d, "softmax");

        logloss<<<ceil(BATCH_SIZE/32.), 32>>>(L, A2_d, y_train_d, OUTPUT_DIM);  
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        //back(1, 10, L, "logloss");

        rLoss<<<ceil(BATCH_SIZE/32.), 32>>>(l, L);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        back(1,1,l, "loss per batch");


        //// backward starts here: 

        dZ<<<gridDimOB,blockDim16>>>(dZ2_d, A2_d, y_train_d, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        // dW2  = A1^T @ dZ2 //db2
        mult_A_T_B<<<gridDimOH, blockDim16>>>(A1_d, dZ2_d, dW2_d, HIDDEN_DIM, BATCH_SIZE, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        db<<<1,OUTPUT_DIM>>>(db2_d, dZ2_d, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

       // dA1 = dZ2 @ W2^T
        mult_A_B_T<<<gridDimHB, blockDim16>>>(dZ2_d, W2_d, dA1_d, BATCH_SIZE,OUTPUT_DIM,HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        dRelu<<<gridDimHB, blockDim16>>>(dA1_d, Z1_d, dZ1_d, HIDDEN_DIM);
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_KERNEL();

        // dW1 = X^T @ dZ1 [I,B] @ [B, H] == [I,H] >> (16,49)
        mult_A_T_B<<<gridDimHI, blockDim16>>>(X_train_d, dZ1_d, dW1_d, INPUT_DIM, BATCH_SIZE, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        db<<<1, HIDDEN_DIM>>>(db1_d, dZ1_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        update2D<<<gridDimHI,blockDim16>>>(W1_d, dW1_d, INPUT_DIM,  HIDDEN_DIM);
        update1D<<<1,HIDDEN_DIM>>>(b1_d, db1_d, HIDDEN_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        update2D<<<gridDimOH,blockDim16>>>(W2_d, dW2_d, HIDDEN_DIM, OUTPUT_DIM);
        update1D<<<1,INPUT_DIM>>>(b2_d, db2_d, OUTPUT_DIM);
        CHECK_KERNEL();
        CHECK_CUDA(cudaDeviceSynchronize());

        //cudaError_t err = cudaGetLastError();
        ////if (err != cudaSuccess) {
        ////    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << "\n";
        ////    return;
        ////}



        //back(INPUT_DIM, HIDDEN_DIM, dW1_d, "dW1: ");
        //back(1, HIDDEN_DIM, b1_d, "b1: ");

    }

    return 0;
}
