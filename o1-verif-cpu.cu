#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <fstream>
#include <algorithm>

// Suppose these come from your project:
#include "utils.hpp"     // for read_mnist_data, xavier_init, etc.

// ========== Constants (match GPU code) ========== //
static const int IMAGE_SIZE  = 784;     // 28x28
static const int NUM_IMAGES  = 60000;
static const int INPUT_DIM   = 784;
static const int HIDDEN_DIM  = 10;
static const int BATCH_SIZE  = 64;

// ========== Forward pass ========== //
//   Z = X*W + b
//   A = softmax(Z)
//   returns mean cross-entropy loss
float forward_cpu(
    const std::vector<float>& X,   // [B, inDim]
    const std::vector<float>& W,   // [inDim, outDim]
    const std::vector<float>& b,   // [outDim]
    const std::vector<float>& y,   // [B] labels in [0..outDim)
    int B, int inDim, int outDim,
    std::vector<float>& Z,         // [B, outDim]
    std::vector<float>& A          // [B, outDim]
)
{
    // 1) Z = X * W
    for(int i = 0; i < B; ++i) {
        for(int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for(int k = 0; k < inDim; ++k) {
                sumVal += X[i*inDim + k] * W[k*outDim + j];
            }
            Z[i*outDim + j] = sumVal;
        }
    }

    // 2) add b
    for(int i = 0; i < B; ++i) {
        for(int j = 0; j < outDim; ++j) {
            Z[i*outDim + j] += b[j];
        }
    }

    // 3) softmax
    for(int i = 0; i < B; ++i) {
        // find max for numerical stability
        float maxVal = Z[i*outDim + 0];
        for(int j = 1; j < outDim; ++j) {
            float v = Z[i*outDim + j];
            if(v > maxVal) maxVal = v;
        }
        // sum of exp
        float sumExp = 0.f;
        for(int j = 0; j < outDim; ++j) {
            float e = std::exp(Z[i*outDim + j] - maxVal);
            A[i*outDim + j] = e;
            sumExp += e;
        }
        // normalize
        for(int j = 0; j < outDim; ++j) {
            A[i*outDim + j] /= sumExp;
        }
    }


    // 4) cross-entropy loss (mean over batch)
    float totalLoss = 0.f;
    for(int i = 0; i < B; ++i) {
        int label = static_cast<int>(y[i]);
        // clamp to avoid log(0)
        float pred = std::max(A[i*outDim + label], 1e-30f);
        totalLoss += -std::log(pred);
    }
    return totalLoss / float(B); 
}

// ========== Backward pass ========== //
//   dZ = A - one_hot(labels)
//   dW = X^T * dZ
//   db = sum(dZ)
//   (here we average over B in dZ)
void backward_cpu(
    const std::vector<float>& X,     // [B, inDim]
    const std::vector<float>& A,     // [B, outDim]
    const std::vector<float>& y,     // [B]
    int B, int inDim, int outDim,
    std::vector<float>& dW,          // out: [inDim, outDim]
    std::vector<float>& db,          // out: [outDim]
    float lr,                        // learning rate
    std::vector<float>& W,          // in-place update
    std::vector<float>& b           // in-place update
)
{
    // 1) dZ
    //  We'll store the gradient in-place in a temporary array.
    std::vector<float> dZ(B * outDim, 0.f);
    for(int i = 0; i < B; ++i) {
        int label = static_cast<int>(y[i]);
        for(int j = 0; j < outDim; ++j) {
            float grad = A[i*outDim + j];
            if(j == label) {
                grad -= 1.f;
            }
            // average across batch
            dZ[i*outDim + j] = grad / float(B);
        }
    }



    // 2) dW = X^T * dZ
    //    shape => [inDim, outDim]
    std::fill(dW.begin(), dW.end(), 0.f);
    for(int k = 0; k < inDim; ++k) {
        for(int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for(int i = 0; i < B; ++i) {
                sumVal += X[i*inDim + k] * dZ[i*outDim + j];
            }
            dW[k*outDim + j] = sumVal;
        }
    }

    std::cout <<"\n dw\n"; // W1 is input_dim x hidden_dim
    for (int i=0; i < inDim; i++){
        for (int j=0; j < outDim; j++){
            std::cout << dW[i*outDim+j] << " ";
            
        }
        std::cout << "\n";
    }
    std::cout <<"dw-DONE \n"; // W1 is input_dim x hidden_dim


    // 3) db = sum(dZ) along batch => shape [outDim]
    std::fill(db.begin(), db.end(), 0.f);
    for(int j = 0; j < outDim; ++j) {
        float sumVal = 0.f;
        for(int i = 0; i < B; ++i) {
            sumVal += dZ[i*outDim + j];
        }
        db[j] = sumVal;
    }


    std::cout <<"\n db\n"; // W1 is input_dim x hidden_dim
    for (int j=0; j < 10; j++){
        std::cout << db[j] << " ";
    }
    std::cout << "\n";
    std::cout <<"db-DONE \n"; // W1 is input_dim x hidden_dim

    // 4) update in-place
    for(int idx = 0; idx < (inDim * outDim); ++idx) {
        W[idx] -= lr * dW[idx];
    }
    for(int j = 0; j < outDim; ++j) {
        b[j] -= lr * db[j];
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

static void forward_cpu_inference(
    const std::vector<float>& X,  // [B, inDim]
    const std::vector<float>& W,  // [inDim, outDim]
    const std::vector<float>& b,  // [outDim]
    int B, int inDim, int outDim,
    std::vector<float>& A         // out: [B, outDim] (softmax)
) {
    // We'll need a temporary array Z = XW + b
    std::vector<float> Z(B * outDim, 0.0f);

    // 1) Z = X * W
    for(int i = 0; i < B; ++i) {
        for(int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for(int k = 0; k < inDim; ++k) {
                sumVal += X[i*inDim + k] * W[k*outDim + j];
            }
            Z[i*outDim + j] = sumVal;
        }
    }

    // 2) Add bias b
    for(int i = 0; i < B; ++i) {
        for(int j = 0; j < outDim; ++j) {
            Z[i*outDim + j] += b[j];
        }
    }

    // 3) Softmax row by row
    for(int i = 0; i < B; ++i) {
        // find max for numerical stability
        float maxVal = Z[i*outDim];
        for(int j = 1; j < outDim; ++j) {
            float val = Z[i*outDim + j];
            if(val > maxVal) {
                maxVal = val;
            }
        }
        // sum of exp
        float sumExp = 0.f;
        for(int j = 0; j < outDim; ++j) {
            float e = std::exp(Z[i*outDim + j] - maxVal);
            A[i*outDim + j] = e;
            sumExp += e;
        }
        // normalize
        for(int j = 0; j < outDim; ++j) {
            A[i*outDim + j] /= sumExp;
        }
    }
}

//--------------------------------------------------------------------
// test_model_cpu
//   - loads test data from disk
//   - runs forward pass in mini-batches
//   - computes final accuracy (0..1)
//--------------------------------------------------------------------
float test_model_cpu(
    const std::vector<float>& W,  // [inDim, outDim]
    const std::vector<float>& b,  // [outDim]
    const std::string& testImagesPath,   // e.g. "data/test_mnist_images.bin"
    const std::string& testLabelsPath,   // e.g. "data/test_mnist_labels.bin"
    int numTestImages,                  // e.g. 10000
    int imageSize,                      // e.g. 784
    int batchSize,                      // e.g. 64
    int inDim,                          // e.g. 784
    int outDim                          // e.g. 10
)
{
    // 1) Read the entire test set
    //    X_test => [10000*784], y_test => [10000]
    std::vector<float> X_test(numTestImages * imageSize);
    std::vector<float> y_test(numTestImages);

    bool ok = read_mnist_data(
        testImagesPath,
        testLabelsPath,
        X_test,
        y_test,
        numTestImages,
        imageSize
    );
    if(!ok) {
        std::cerr << "Error reading test data.\n";
        return -1.0f;
    }

    // 2) Evaluate in mini-batches to avoid using too much memory at once
    int numBatches = numTestImages / batchSize;  // ignoring leftover for simplicity

    int totalCorrect = 0;
    int totalCount   = 0;

    // We'll allocate a buffer for the forward pass output
    std::vector<float> A(batchSize * outDim, 0.f);

    for(int bIdx = 0; bIdx < numBatches; ++bIdx) {
        int startIdx = bIdx * batchSize;
        // slice out the batch
        std::vector<float> X_batch(batchSize * inDim);
        std::vector<float> y_batch(batchSize);

        for(int i = 0; i < batchSize; ++i) {
            int idxGlobal = startIdx + i;
            // copy the image
            std::copy_n(
                X_test.begin() + idxGlobal * imageSize,
                imageSize,
                X_batch.begin() + i*inDim
            );
            // copy label
            y_batch[i] = y_test[idxGlobal];
        }

        // 3) Forward pass (inference)
        forward_cpu_inference(X_batch, W, b, batchSize, inDim, outDim, A);

        // 4) Determine predictions & count accuracy
        for(int i = 0; i < batchSize; ++i) {
            // find argmax in A[i]
            int predLabel = 0;
            float maxProb = A[i*outDim + 0];
            for(int j = 1; j < outDim; ++j) {
                float val = A[i*outDim + j];
                if(val > maxProb) {
                    maxProb = val;
                    predLabel = j;
                }
            }
            int trueLabel = static_cast<int>(y_batch[i]);
            if(predLabel == trueLabel) {
                totalCorrect++;
            }
        }

        totalCount += batchSize;
    }

    float accuracy = float(totalCorrect) / float(totalCount);
    return accuracy;
}



int main() {
    // ------------------------------------------------------------
    // 1) Read MNIST data
    // ------------------------------------------------------------
    std::vector<float> X_train(NUM_IMAGES * IMAGE_SIZE), y_train(NUM_IMAGES);
    if (!read_mnist_data("data/train_mnist_images.bin",
                         "data/train_mnist_labels.bin",
                          X_train,
                          y_train,
                          NUM_IMAGES,
                          IMAGE_SIZE))
    {
        std::cerr << "Error: could not read MNIST.\n";
        return -1;
    }

    // ------------------------------------------------------------
    // 2) Initialize W, b with Xavier (same as GPU code)
    // ------------------------------------------------------------
    std::vector<float> W(INPUT_DIM * HIDDEN_DIM);
    std::vector<float> b(HIDDEN_DIM);
    utils::xavier_init(W.data(), b.data(), INPUT_DIM, HIDDEN_DIM);

    for (int i=0; i < 10; i++){
        for (int j=0; j < 5; j++){
            std::cout << W[i*HIDDEN_DIM+j] << " ";
        }
        std::cout << "\n";
    }

    // ------------------------------------------------------------
    // 3) Loop over mini-batches of size 64
    //    We'll do 1 epoch, ignoring leftover images if 60000
    //    is not divisible by 64 exactly.
    // ------------------------------------------------------------
    int numBatches = NUM_IMAGES / BATCH_SIZE;
    float lr = 0.001f;  // your chosen learning rate

    // Buffers for forward pass
    std::vector<float> Z(BATCH_SIZE * HIDDEN_DIM);
    std::vector<float> A(BATCH_SIZE * HIDDEN_DIM);

    // Buffers for backward pass
    std::vector<float> dW(INPUT_DIM * HIDDEN_DIM);
    std::vector<float> db(HIDDEN_DIM);

    int n_epochs = 1;
    numBatches = 1;

    float testAcc = 0.0f; 
    const std::string testImagesBin = "data/test_mnist_images.bin";
    const std::string testLabelsBin = "data/test_mnist_labels.bin";
    const int NUM_TEST_IMAGES = 10000;

    for(int kk= 0; kk < n_epochs; ++kk) {
        for(int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            std::cout <<"\nBatch =  " << batchIdx << "\n"; // W1 is input_dim x hidden_dim
            // Slice out [batchIdx * 64 .. (batchIdx+1)*64) from X_train, y_train
            const int start = batchIdx * BATCH_SIZE;
            const int end   = start + BATCH_SIZE; // not inclusive

            // Create X_batch, y_batch for this iteration
            // shape: X_batch -> [64, 784], y_batch -> [64]
            std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);
            std::vector<float> y_batch(BATCH_SIZE);

            // copy
            for(int i = 0; i < BATCH_SIZE; ++i) {
                // each sample
                int idxGlobal = start + i;
                // copy 784 floats
                std::copy_n(X_train.begin() + idxGlobal * IMAGE_SIZE,
                            IMAGE_SIZE,
                            X_batch.begin() + i * INPUT_DIM);
                // copy label
                y_batch[i] = y_train[idxGlobal];
            }

            // Forward
            float loss = forward_cpu(X_batch, W, b, y_batch,
                                    BATCH_SIZE, INPUT_DIM, HIDDEN_DIM,
                                    Z, A);


            std::cout <<"\nLoss = " << loss <<"\n"; 

            std::cout <<"\n smx before the update \n"; 
            for (int i=0; i < 4; i++){
                for (int j=0; j < 10; j++){
                    std::cout << A[i*HIDDEN_DIM+j] << " ";
                }
                std::cout << "\n";
            }
            // Backward + Update
            backward_cpu(X_batch, A, y_batch,
                        BATCH_SIZE, INPUT_DIM, HIDDEN_DIM,
                        dW, db, lr,
                        W, b);

            if((batchIdx+1) % 100 == 0) {
                std::cout << "Batch " << (batchIdx+1)
                        << "/" << numBatches
                        << " => Loss = " << loss << "\n";
                testAcc = test_model_cpu(
                                            W,
                                            b,
                                            testImagesBin,
                                            testLabelsBin,
                                            NUM_TEST_IMAGES,
                                            IMAGE_SIZE,
                                            64,           // batchSize for inference
                                            INPUT_DIM,
                                            HIDDEN_DIM
                                        );
                std::cout << "Epoch " << kk 
                        << "Batch " << (batchIdx+1)
                        << "/" << numBatches
                        << " => Loss = " << loss 
                        << "Test Accuracy = " << (testAcc * 100.0f) << "%\n";
            }
        }
    }
    return 0;
}
