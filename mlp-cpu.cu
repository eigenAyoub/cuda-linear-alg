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
static const int HIDDEN_DIM  = 256;      // changed from 10 to 256 for hidden layer
static const int OUTPUT_DIM  = 10;       // final output dimension
static const int BATCH_SIZE  = 64;

// ========== Forward Pass ========== //
// This function computes a two-layer network:
// Layer 1 (Hidden):
//    Z1 = X * W1 + b1       (X: [B,784], W1: [784,256], b1: [256])
//    A1 = ReLU(Z1)
// Layer 2 (Output):
//    Z2 = A1 * W2 + b2       (W2: [256,10], b2: [10])
//    A2 = softmax(Z2)
// Returns the mean cross-entropy loss and also outputs the intermediate
// activations (A1) and final activations (A2). It also stores Z1 if you need it
// in backprop for computing the derivative of ReLU.
float forward_cpu(
    const std::vector<float>& X,         // [B, INPUT_DIM]
    const std::vector<float>& W1,        // [INPUT_DIM, HIDDEN_DIM]
    const std::vector<float>& b1,        // [HIDDEN_DIM]
    const std::vector<float>& W2,        // [HIDDEN_DIM, OUTPUT_DIM]
    const std::vector<float>& b2,        // [OUTPUT_DIM]
    const std::vector<float>& y,         // [B] (labels in [0..OUTPUT_DIM-1])
    int B, int inDim, int hiddenDim, int outDim,
    std::vector<float>& Z1,              // [B, HIDDEN_DIM] pre-ReLU values
    std::vector<float>& A1,              // [B, HIDDEN_DIM] activations (ReLU)
    std::vector<float>& Z2,              // [B, OUTPUT_DIM] logits (pre-softmax)
    std::vector<float>& A2               // [B, OUTPUT_DIM] softmax probabilities
)
{
    // -------- Layer 1: Hidden Layer --------
    // Compute Z1 = X * W1 + b1
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < hiddenDim; ++j) {
            float sumVal = 0.f;
            for (int k = 0; k < inDim; ++k) {
                sumVal += X[i * inDim + k] * W1[k * hiddenDim + j];
            }
            // add bias
            Z1[i * hiddenDim + j] = sumVal + b1[j];
            // ReLU activation: A1 = ReLU(Z1)
            A1[i * hiddenDim + j] = (Z1[i * hiddenDim + j] > 0.f) ? Z1[i * hiddenDim + j] : 0.f;
        }
    }

    // -------- Layer 2: Output Layer --------
    // Compute Z2 = A1 * W2 + b2   where A1: [B, hiddenDim], W2: [hiddenDim, outDim]
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for (int k = 0; k < hiddenDim; ++k) {
                sumVal += A1[i * hiddenDim + k] * W2[k * outDim + j];
            }
            Z2[i * outDim + j] = sumVal + b2[j];
        }
    }

    // Softmax for each sample in the batch (with numerical stability):
    float totalLoss = 0.f;
    for (int i = 0; i < B; ++i) {
        // Find max value in Z2[i] for numerical stability.
        float maxVal = Z2[i * outDim + 0];
        for (int j = 1; j < outDim; ++j) {
            float v = Z2[i * outDim + j];
            if (v > maxVal) maxVal = v;
        }
        // Compute exponentials and sum them.
        float sumExp = 0.f;
        for (int j = 0; j < outDim; ++j) {
            float e = std::exp(Z2[i * outDim + j] - maxVal);
            A2[i * outDim + j] = e;
            sumExp += e;
        }
        // Normalize and also compute cross-entropy loss.
        for (int j = 0; j < outDim; ++j) {
            A2[i * outDim + j] /= sumExp;
        }

        // loss contribution: -log(probability of correct class)
        int label = static_cast<int>(y[i]);
        float pred = std::max(A2[i * outDim + label], 1e-30f);
        totalLoss += -std::log(pred);
    }
    return totalLoss / float(B);
}

// ========== Backward Pass ========== //
// This function computes the gradients for both layers.
// It computes the following:
// 1. Compute delta2 = A2 - one_hot(y)  [gradient at output]
// 2. dW2 = (A1)^T * delta2,  db2 = sum(delta2)
// 3. Backpropagate delta1 = (delta2 * W2^T) * ReLU'(Z1)   (element-wise multiplication)
// 4. dW1 = X^T * delta1,   db1 = sum(delta1)
// 5. Update: W2, b2, W1, b1 with gradient descent (learning rate lr)
void backward_cpu(
    const std::vector<float>& X,         // [B, INPUT_DIM]
    const std::vector<float>& Z1,        // [B, HIDDEN_DIM] (pre-ReLU)
    const std::vector<float>& A1,        // [B, HIDDEN_DIM] (post-ReLU)
    const std::vector<float>& A2,        // [B, OUTPUT_DIM] (softmax)
    const std::vector<float>& y,         // [B]
    int B, int inDim, int hiddenDim, int outDim,
    std::vector<float>& dW1,             // [INPUT_DIM, HIDDEN_DIM]
    std::vector<float>& db1,             // [HIDDEN_DIM]
    std::vector<float>& dW2,             // [HIDDEN_DIM, OUTPUT_DIM]
    std::vector<float>& db2,             // [OUTPUT_DIM]
    float lr,                          // learning rate
    std::vector<float>& W1,            // in-place update
    std::vector<float>& b1,            // in-place update
    std::vector<float>& W2,            // in-place update
    std::vector<float>& b2             // in-place update
)
{
    // --- Step 1: Compute delta2 = A2 - one_hot(y)  (averaged over batch) ---
    std::vector<float> delta2(B * outDim, 0.f);
    for (int i = 0; i < B; ++i) {
        int label = static_cast<int>(y[i]);
        for (int j = 0; j < outDim; ++j) {
            float grad = A2[i * outDim + j];
            if(j == label)
                grad -= 1.f;
            // Average over the batch
            delta2[i * outDim + j] = grad / float(B);
        }
    }

    // --- Step 2: Gradients for Layer 2 ---
    // dW2 = A1^T * delta2. A1: [B, hiddenDim], delta2: [B, outDim] => dW2: [hiddenDim, outDim]
    std::fill(dW2.begin(), dW2.end(), 0.f);
    for (int k = 0; k < hiddenDim; ++k) {
        for (int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for (int i = 0; i < B; ++i) {
                sumVal += A1[i * hiddenDim + k] * delta2[i * outDim + j];
            }
            dW2[k * outDim + j] = sumVal;
        }
    }
    // db2 = sum(delta2) along batch, for each output unit
    std::fill(db2.begin(), db2.end(), 0.f);
    for (int j = 0; j < outDim; ++j) {
        float sumVal = 0.f;
        for (int i = 0; i < B; ++i) {
            sumVal += delta2[i * outDim + j];
        }
        db2[j] = sumVal;
    }

    // --- Step 3: Backpropagate to hidden layer ---
    // First, compute dA1 = delta2 * W2^T.  delta2: [B, outDim], W2: [hiddenDim, outDim],
    // so W2^T: [outDim, hiddenDim] => dA1: [B, hiddenDim]
    std::vector<float> dA1(B * hiddenDim, 0.f);
    for (int i = 0; i < B; ++i) {
        for (int k = 0; k < hiddenDim; ++k) {
            float sumVal = 0.f;
            for (int j = 0; j < outDim; ++j) {
                sumVal += delta2[i * outDim + j] * W2[k * outDim + j];
            }
            dA1[i * hiddenDim + k] = sumVal;
        }
    }
    
    // Multiply by ReLU derivative to get delta1.
    // delta1 = dA1 .* ReLU'(Z1)
    std::vector<float> delta1(B * hiddenDim, 0.f);
    for (int i = 0; i < B; ++i) {
        for (int k = 0; k < hiddenDim; ++k) {
            float relu_deriv = (Z1[i * hiddenDim + k] > 0.f) ? 1.f : 0.f;
            delta1[i * hiddenDim + k] = dA1[i * hiddenDim + k] * relu_deriv;
        }
    }

    // --- Step 4: Gradients for Layer 1 ---
    // dW1 = X^T * delta1. X: [B, inDim], delta1: [B, hiddenDim]
    std::fill(dW1.begin(), dW1.end(), 0.f);
    for (int k = 0; k < inDim; ++k) {
        for (int j = 0; j < hiddenDim; ++j) {
            float sumVal = 0.f;
            for (int i = 0; i < B; ++i) {
                sumVal += X[i * inDim + k] * delta1[i * hiddenDim + j];
            }
            dW1[k * hiddenDim + j] = sumVal;
        }
    }
    // db1 = sum(delta1) along batch, for each hidden unit
    std::fill(db1.begin(), db1.end(), 0.f);
    for (int j = 0; j < hiddenDim; ++j) {
        float sumVal = 0.f;
        for (int i = 0; i < B; ++i) {
            sumVal += delta1[i * hiddenDim + j];
        }
        db1[j] = sumVal;
    }
    
    // --- Step 5: Update Parameters ---
    // Update Layer 2
    for (int idx = 0; idx < (hiddenDim * outDim); ++idx) {
        W2[idx] -= lr * dW2[idx];
    }
    for (int j = 0; j < outDim; ++j) {
        b2[j] -= lr * db2[j];
    }

    // Update Layer 1
    for (int idx = 0; idx < (inDim * hiddenDim); ++idx) {
        W1[idx] -= lr * dW1[idx];
    }
    for (int j = 0; j < hiddenDim; ++j) {
        b1[j] -= lr * db1[j];
    }
}


// ========== MNIST Data Reading ==========
bool read_mnist_data(
    const std::string& images_path,
    const std::string& labels_path,
    std::vector<float>& images,
    std::vector<float>& labels,
    const int num_images,
    const int image_size   
) {
    std::ifstream images_file(images_path, std::ios::binary);
    std::ifstream labels_file(labels_path, std::ios::binary);

    if (!images_file || !labels_file) {
        std::cerr << "Error opening MNIST files" << std::endl;
        return false;
    }

    std::vector<uint8_t> images_buff(num_images * image_size);
    std::vector<uint8_t> labels_buff(num_images);

    images_file.read(reinterpret_cast<char*>(images_buff.data()), num_images * image_size);
    labels_file.read(reinterpret_cast<char*>(labels_buff.data()), num_images);

    images.resize(num_images * image_size);
    labels.resize(num_images);

    std::copy(images_buff.begin(), images_buff.end(), images.begin());
    std::copy(labels_buff.begin(), labels_buff.end(), labels.begin());

    return true;
}


// ========== Forward Inference (for testing) ==========
static void forward_cpu_inference(
    const std::vector<float>& X,  // [B, INPUT_DIM]
    const std::vector<float>& W1, // [INPUT_DIM, HIDDEN_DIM]
    const std::vector<float>& b1, // [HIDDEN_DIM]
    const std::vector<float>& W2, // [HIDDEN_DIM, OUTPUT_DIM]
    const std::vector<float>& b2, // [OUTPUT_DIM]
    int B, int inDim, int hiddenDim, int outDim,
    std::vector<float>& A2         // out: [B, OUTPUT_DIM] (softmax)
) {
    // Temporary buffers for hidden layer & output layer
    std::vector<float> Z1(B * hiddenDim, 0.f);
    std::vector<float> A1(B * hiddenDim, 0.f);
    std::vector<float> Z2(B * outDim, 0.f);

    // Layer 1: Z1 = X * W1 + b1, A1 = ReLU(Z1)
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < hiddenDim; ++j) {
            float sumVal = 0.f;
            for (int k = 0; k < inDim; ++k) {
                sumVal += X[i * inDim + k] * W1[k * hiddenDim + j];
            }
            Z1[i * hiddenDim + j] = sumVal + b1[j];
            A1[i * hiddenDim + j] = (Z1[i * hiddenDim + j] > 0.f) ? Z1[i * hiddenDim + j] : 0.f;
        }
    }

    // Layer 2: Z2 = A1 * W2 + b2
    for (int i = 0; i < B; ++i) {
        for (int j = 0; j < outDim; ++j) {
            float sumVal = 0.f;
            for (int k = 0; k < hiddenDim; ++k) {
                sumVal += A1[i * hiddenDim + k] * W2[k * outDim + j];
            }
            Z2[i * outDim + j] = sumVal + b2[j];
        }
    }

    // Softmax on Z2
    for (int i = 0; i < B; ++i) {
        float maxVal = Z2[i * outDim + 0];
        for (int j = 1; j < outDim; ++j) {
            if(Z2[i*outDim+j] > maxVal) maxVal = Z2[i*outDim+j];
        }
        float sumExp = 0.f;
        for (int j = 0; j < outDim; ++j) {
            float e = std::exp(Z2[i * outDim + j] - maxVal);
            A2[i * outDim + j] = e;
            sumExp += e;
        }
        for (int j = 0; j < outDim; ++j) {
            A2[i * outDim + j] /= sumExp;
        }
    }
}


// ========== Testing the Model (CPU Inference) ==========
float test_model_cpu(
    const std::vector<float>& W1,  // [INPUT_DIM, HIDDEN_DIM]
    const std::vector<float>& b1,  // [HIDDEN_DIM]
    const std::vector<float>& W2,  // [HIDDEN_DIM, OUTPUT_DIM]
    const std::vector<float>& b2,  // [OUTPUT_DIM]
    const std::string& testImagesPath,
    const std::string& testLabelsPath,
    int numTestImages,
    int imageSize,
    int batchSize,
    int inDim,
    int hiddenDim,
    int outDim
)
{
    std::vector<float> X_test(numTestImages * imageSize);
    std::vector<float> y_test(numTestImages);

    bool ok = read_mnist_data(testImagesPath, testLabelsPath, X_test, y_test, numTestImages, imageSize);
    if (!ok) {
        std::cerr << "Error reading test data.\n";
        return -1.0f;
    }

    int numBatches = numTestImages / batchSize;
    int totalCorrect = 0;
    int totalCount = 0;
    std::vector<float> A2(batchSize * outDim, 0.f);

    for (int bIdx = 0; bIdx < numBatches; ++bIdx) {
        int startIdx = bIdx * batchSize;
        std::vector<float> X_batch(batchSize * inDim);
        std::vector<float> y_batch(batchSize);

        for (int i = 0; i < batchSize; ++i) {
            int idxGlobal = startIdx + i;
            std::copy_n(X_test.begin() + idxGlobal * imageSize, imageSize, X_batch.begin() + i * inDim);
            y_batch[i] = y_test[idxGlobal];
        }

        forward_cpu_inference(X_batch, W1, b1, W2, b2, batchSize, inDim, hiddenDim, outDim, A2);

        for (int i = 0; i < batchSize; ++i) {
            int predLabel = 0;
            float maxProb = A2[i * outDim + 0];
            for (int j = 1; j < outDim; ++j) {
                if (A2[i * outDim + j] > maxProb) {
                    maxProb = A2[i * outDim + j];
                    predLabel = j;
                }
            }
            int trueLabel = static_cast<int>(y_batch[i]);
            if (predLabel == trueLabel)
                totalCorrect++;
        }
        totalCount += batchSize;
    }

    float accuracy = float(totalCorrect) / float(totalCount);
    return accuracy;
}


// ========== Main ==========
int main() {
    // --- 1) Read MNIST training data ---
    std::vector<float> X_train(NUM_IMAGES * IMAGE_SIZE);
    std::vector<float> y_train(NUM_IMAGES);
    if (!read_mnist_data("data/train_mnist_images.bin", "data/train_mnist_labels.bin",
                         X_train, y_train, NUM_IMAGES, IMAGE_SIZE)) {
        std::cerr << "Error: could not read MNIST training data.\n";
        return -1;
    }

    // --- 2) Initialize parameters ---
    // Layer 1: Hidden layer parameters: W1 and b1
    std::vector<float> W1(INPUT_DIM * HIDDEN_DIM);
    std::vector<float> b1(HIDDEN_DIM);
    // Layer 2: Output layer parameters: W2 and b2
    std::vector<float> W2(HIDDEN_DIM * OUTPUT_DIM);
    std::vector<float> b2(OUTPUT_DIM);
    
    // Xavier initialization (utils::xavier_init should be adapted to initialize both layers)
    // Here we assume it can initialize a weight matrix and bias vector.
    utils::xavier_init(W1.data(), b1.data(), INPUT_DIM, HIDDEN_DIM);
    utils::xavier_init(W2.data(), b2.data(), HIDDEN_DIM, OUTPUT_DIM);
    
    // --- 3) Set training hyperparameters ---
    int numBatches = NUM_IMAGES / BATCH_SIZE; // ignoring leftover
    float lr = 0.0001f;  // learning rate
    int n_epochs = 1;
    // For demonstration, we can run a limited number of batches, e.g.:
    
    // Buffers for forward and backward pass for layer1 and layer2.
    // Layer 1 buffers:
    std::vector<float> Z1(BATCH_SIZE * HIDDEN_DIM, 0.f);
    std::vector<float> A1(BATCH_SIZE * HIDDEN_DIM, 0.f);
    // Layer 2 buffers:
    std::vector<float> Z2(BATCH_SIZE * OUTPUT_DIM, 0.f);
    std::vector<float> A2(BATCH_SIZE * OUTPUT_DIM, 0.f);
    
    // Gradient buffers for both layers:
    std::vector<float> dW1(INPUT_DIM * HIDDEN_DIM, 0.f);
    std::vector<float> db1(HIDDEN_DIM, 0.f);
    std::vector<float> dW2(HIDDEN_DIM * OUTPUT_DIM, 0.f);
    std::vector<float> db2(OUTPUT_DIM, 0.f);
    
    const std::string testImagesBin = "data/test_mnist_images.bin";
    const std::string testLabelsBin = "data/test_mnist_labels.bin";
    const int NUM_TEST_IMAGES = 10000;
    float testAcc = 0.0f;

    numBatches = 2;
    
    // --- 4) Training Loop (1 epoch) ---
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        for (int batchIdx = 0; batchIdx < numBatches; ++batchIdx) {
            std::cout << "\nEpoch " << epoch+1 << " Batch " << batchIdx+1 << "\n";
            int start = batchIdx * BATCH_SIZE;
            int end = start + BATCH_SIZE;
            
            // Prepare mini-batch
            std::vector<float> X_batch(BATCH_SIZE * INPUT_DIM);
            std::vector<float> y_batch(BATCH_SIZE);
            for (int i = 0; i < BATCH_SIZE; ++i) {
                int idxGlobal = start + i;
                std::copy_n(X_train.begin() + idxGlobal * IMAGE_SIZE,
                            IMAGE_SIZE,
                            X_batch.begin() + i * INPUT_DIM);
                y_batch[i] = y_train[idxGlobal];
            }
            
            // --- Forward Pass ---
            // Layer 1 and Layer 2 forward:
            float loss = forward_cpu(X_batch, W1, b1, W2, b2,
                                     y_batch,
                                     BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                                     Z1, A1, Z2, A2);
            std::cout << "Loss: " << loss << "\n";
            
            // --- Backward Pass ---
            backward_cpu(X_batch, Z1, A1, A2, y_batch,
                         BATCH_SIZE, INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM,
                         dW1, db1, dW2, db2, lr,
                         W1, b1, W2, b2);
            
            // Every 100 batches (or here, for our demo, at the end) test the model:
            if ((batchIdx + 1) % 100 == 0 || (batchIdx + 1) == numBatches) {
                testAcc = test_model_cpu(W1, b1, W2, b2,
                                         testImagesBin,
                                         testLabelsBin,
                                         NUM_TEST_IMAGES,
                                         IMAGE_SIZE,
                                         BATCH_SIZE,
                                         INPUT_DIM,
                                         HIDDEN_DIM,
                                         OUTPUT_DIM);
                std::cout << "Test Accuracy: " << (testAcc * 100.0f) << "%\n";
            }
        }
    }
    return 0;
}
