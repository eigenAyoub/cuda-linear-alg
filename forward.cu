#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

const int IMAGE_SIZE  = 784;
const int NUM_IMAGES  = 60000;
const int IMAGE_WIDTH = 28;

const int INPUT_DIM  = 784;
const int OUTPUT_DIM = 10;
const int HIDDEM_DIM = 10;

const int BATCH_SIZE = 1024;

int main(){

    std::ifstream X_train_file("data/train_mnist_images.bin", std::ios::binary);
    std::ifstream y_train_file("data/train_mnist_labels.bin", std::ios::binary);

    if (!X_train_file || !y_train_file){
        std::cout << "Oupsie" << "\n";
        return -1;
    }

    std::vector<uint8_t> X_train_buff(NUM_IMAGES*IMAGE_SIZE);
    std::vector<uint8_t> y_train_buff(NUM_IMAGES);

    std::vector<float> X_train(NUM_IMAGES*IMAGE_SIZE);
    std::vector<float> y_train(NUM_IMAGES);

    // ignore multiplying by `sizeof(uint8_t)` as it's = 1
    X_train_file.read(reinterpret_cast<char *>(X_train_buff.data()), IMAGE_SIZE*NUM_IMAGES);
    y_train_file.read(reinterpret_cast<char *>(y_train_buff.data()), NUM_IMAGES);

    std::copy(X_train_buff.begin(), X_train_buff.end(), X_train.begin());
    std::copy(y_train_buff.begin(), y_train_buff.end(), y_train.begin());


    std::vector<float> W1_h(INPUT_DIM*HIDDEM_DIM);
    std::vector<float> b1_h(HIDDEM_DIM);
    //std::vector<float> Y1_h(BATCH_SIZE*HIDDEM_DIM); // Y1_h = X @ W1_h
    //std::vector<float> Z1_h(BATCH_SIZE*HIDDEM_DIM); // Z1_h = Y1_h + b1_h 
    //std::vector<float> A1_h(BATCH_SIZE*HIDDEM_DIM); // A1_h = relu(Z1_h)

    float * W1_d;
    float * b1_d;
    float * Y1_d;  // Y1_h = X @ W1_h
    float * Z1_d;  // Z1_h = Y1_h + b1_h 
    float * A1_d;  // A1_h = relu(Z1_h)
    float * loss;  // loss =  Σ - q_i log(A_i)  // sum over samples?
    float * rloss; // A1_h = relu(Z1_h)

    cudaMalloc((void **) &W1_d, sizeof(float)*W1_h.size());
    cudaMalloc((void **) &b1_d, sizeof(float)*b1_h.size());
    cudaMalloc((void **) &Y1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);
    cudaMalloc((void **) &Z1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);
    cudaMalloc((void **) &A1_d, sizeof(float)*BATCH_SIZE*HIDDEM_DIM);

    cudaMemcpy(W1_d, W1_h.data(), W1_h.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(b1_d, W1_h.data(), b1_h.size()*sizeof(float), cudaMemcpyHostToDevice);


    dim3 tpb1d(256); // as thread per block linear
    dim3 tpb2d(64, 8); // as thread per block linear // 512 per block

    dim3 numBx1d((BATCH_SIZE + tpb1d.x - 1) / tpb1d.x);
    dim3 numBx2d(std::ceil(BATCH_SIZE/tpb2d.x), std::ceil(HIDDEM_DIM/tpb1d.y));

    //rSoftmax<<<numBlocks, tpbl>>>(S, true_y, rS, B, C);

    return 0;
}