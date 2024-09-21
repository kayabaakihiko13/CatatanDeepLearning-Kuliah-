#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "utils.h"
typedef struct {
    int inputSize;
    int hiddenSize;
    int outputSize;
    float *hiddenWeight;
    float *outputHidden;
    float *hiddenBias;
    float *outputBias;
} NeuralNetwork;
void ZScoreNorm(Matrixf *mat){
    // calculate mean
    float sum=0.0f;
    int totalElements = mat->rows*mat->cols;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            sum += mat->data[i * mat->cols + j];
        }
    }
    float mean = sum / totalElements; 
    float varianceSum = 0.0f;
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            float diff = mat->data[i * mat->cols + j] - mean;
            varianceSum += diff * diff;
        }
    }
    float stddev = sqrt(varianceSum / totalElements);

    // 3. Apply Z-Score normalization
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i * mat->cols + j] = (mat->data[i * mat->cols + j] - mean) / stddev;
        }
    }
}
static Matrixf GaussianKernel(int kernelSize) {
    if (kernelSize % 2 == 0) {
        printf("Error: Kernel size should be odd\n");
        exit(1);
    }

    Matrixf kernel = {kernelSize, kernelSize, NULL};
    kernel.data = (float*)malloc(kernelSize * kernelSize * sizeof(float));
    if (kernel.data == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    float center = (float)(kernelSize - 1) / 2.0f;
    float sigma = 0.3f * ((float)kernelSize - 1.0f) * 0.5f - 1.0f + 0.8f;
    float sum = 0.0f;

    for (int i = 0; i < kernelSize; i++) {
        for (int j = 0; j < kernelSize; j++) {
            float x = i - center;
            float y = j - center;
            float value = expf(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel.data[i * kernelSize + j] = value;
            sum += value;
        }
    }

    // Normalize the kernel
    for (int i = 0; i < kernelSize * kernelSize; i++) {
        kernel.data[i] /= sum;
    }
    return kernel;
}

void applyGaussianFilter(Matrixf*mat,Matrixf*kernel){
    int inputRows=mat->rows;
    int inputCols = mat->cols;
    int kernelSize = kernel->rows;
    int padding = kernelSize/2;
    // Create a temporary matrix to store the result
    Matrixf result = {inputRows, inputCols, NULL};
    result.data = (float*)malloc(inputRows * inputCols * sizeof(float));
    if (result.data == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    // update value in matrix input
    for(int i =0;i<inputRows;i++){
        for(int j=0;j<inputCols;j++){
            float sum = 0.0f;
            for(int ki =-padding;ki<=padding;ki++){
                for(int kj=-padding;kj<=padding;kj++){
                    int ii = i+ki;
                    int jj = j+kj;
                    // check bounds
                    if (ii >= 0 && ii < inputRows && jj >= 0 && jj < inputCols) {
                        float inputVal = mat->data[ii * inputCols + jj];
                        float kernelVal = kernel->data[(ki + padding) * kernelSize + (kj + padding)];
                        sum += inputVal * kernelVal;
                    }
                }
            }
        }
    } 

}

// making activation function
static float relu(float x){
    return (x>0)?x:0;
} 
static float reluDerivate(float x){
    return (x>0)?1:0;
}
static float sigmoid(float x){
    return 1.0 / (1.0 + exp(-x)); 
}


static void softmax(float* input,int size){
    float max = input[0];
    for(int i =0;i<size;i++){
        if(input[i]>max){
            max = input[i];
        }
    }
    float sum = 0.0f;
    for(int i =0;i<size;i++){
        input[i] = exp(input[i]-max);
        sum+=input[i];
    }
    for(int i = 0;i<size;i++){
        input[i] /=sum;
    }
}

// intialization for neural network
NeuralNetwork initializationNetwork(int inputSize,int hiddenSize,int outputSize){
    NeuralNetwork nn;
    nn.inputSize = inputSize;
    nn.hiddenSize = hiddenSize;
    nn.outputSize = outputSize;
    // allocation memory for weight
    nn.hiddenWeight = (float*)malloc(hiddenSize*inputSize * sizeof(float));
    nn.outputHidden = (float*)malloc(outputSize*hiddenSize*sizeof(float));
    nn.hiddenBias = (float*)malloc(hiddenSize*sizeof(float));
    nn.outputBias = (float*)malloc(outputSize*sizeof(float));
    if (nn.hiddenWeight == NULL || nn.outputHidden == NULL || nn.hiddenBias == NULL || nn.outputBias == NULL) {
    // Handle memory allocation failure, free any already allocated memory
    free(nn.hiddenWeight);
    free(nn.outputHidden);
    free(nn.hiddenBias);
    free(nn.outputBias);
    // Consider logging or returning an error code if needed
    }
    // initial for input Weight
    for(int i =0;i<inputSize;i++){
        for(int j=0;j<hiddenSize;j++){
            nn.hiddenWeight[i*inputSize+j] = (float)((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    // initial for outputHidden
    for(int i =0;i<outputSize;i++){
        for(int j =0;j<hiddenSize;j++){
            nn.outputHidden[i*inputSize+j] = (float)((double)rand() / RAND_MAX) * 2 - 1;
        }
    }
    // Initialize biases
    for (int i = 0; i < hiddenSize; i++) {
        nn.hiddenBias[i] = 0.0f;
    }
    for (int i = 0; i < outputSize; i++) {
        nn.outputBias[i] = 0.0f;
    }
    return nn;
}

// Perform forward progration throught the network
void forwardPropagation(NeuralNetwork* nn, float* input, float* hiddenLayer, float* outputLayer) {
    // Inisialisasi hidden layer dengan 0
    #pragma omp parallel for
    for (int i = 0; i < nn->hiddenSize; i++) {
        hiddenLayer[i] = 0.0;
        for (int j = 0; j < nn->inputSize; j++) {
            // Menghitung nilai dot product antara input dan bobot
            hiddenLayer[i] += input[j] * nn->hiddenWeight[i * nn->inputSize + j];
        }
        hiddenLayer[i] += nn->hiddenBias[i];

        // Aktivasi (misal menggunakan sigmoid atau ReLU)
        hiddenLayer[i] = 1.0 / (1.0 + exp(-hiddenLayer[i])); 
    }

    // Inisialisasi output layer
    #pragma omp parallel for
    for (int i = 0; i < nn->outputSize; i++) {
        outputLayer[i] = 0.0;
        for (int j = 0; j < nn->hiddenSize; j++) {
            // Menghitung dot product antara hidden layer dan output bobot
            outputLayer[i] += hiddenLayer[j] * nn->outputHidden[i * nn->hiddenSize + j];
        }
        outputLayer[i] += nn->outputBias[i];
        outputLayer[i] = 1.0 / (1.0 + exp(-outputLayer[i]));  // Sigmoid activation
    }
}

void backwardPropagation(NeuralNetwork *nn, Matrixf *input, float *hiddenLayer, float *outputLayer, float label, float learningRate) {
    float outputError[nn->outputSize];
    float hiddenError[nn->hiddenSize];

    // Calculate output error
    #pragma omp parallel for
    for (int i = 0; i < nn->outputSize; i++) {
        outputError[i] = outputLayer[i] - label;
    }

    // Update weights and biases for output layer
    #pragma omp parallel for
    for (int i = 0; i < nn->outputSize; i++) {
        for (int j = 0; j < nn->hiddenSize; j++) {
            nn->outputHidden[i * nn->hiddenSize + j] -= learningRate * outputError[i] * hiddenLayer[j];
        }
        nn->outputBias[i] -= learningRate * outputError[i];
    }

    // Calculate hidden error
    #pragma omp parallel for
    for (int i = 0; i < nn->hiddenSize; i++) {
        hiddenError[i] = 0.0f;

        for (int j = 0; j < nn->outputSize; j++) {
            hiddenError[i] += nn->outputHidden[j * nn->hiddenSize + i] * outputError[j];
        }

        hiddenError[i] *= relu(hiddenLayer[i]);
    }

    // Update weights and biases for hidden layer
    #pragma omp parallel for
    for (int i = 0; i < nn->hiddenSize; i++) {
        for (int j = 0; j < nn->inputSize; j++) {
            nn->hiddenWeight[i * nn->inputSize + j] -= learningRate * hiddenError[i] * input->data[j];
        }
        nn->hiddenBias[i] -= learningRate * hiddenError[i];
    }
}

void trainNetwork(NeuralNetwork *nn,Matrixf* trainingData,Matrixf* labels,int numSample,int epochs,float learningRate){
    int inputSize = nn->inputSize;
    int hiddenSize = nn->hiddenSize;
    int outputSize = nn->outputSize;

    for(int epoch=0;epoch<epochs;epoch++){
        float totalLoss = 0.0f;
        #pragma omp parallel reduction(+:totalLoss){
        for (int sample = 0; sample < numSample; sample++) {
            // Allocate memory for hidden and output layers
            float hiddenLayer[hiddenSize];
            float outputLayer[outputSize];
            // setup kernel
            Matrixf Kernel = GaussianKernel(3);
            Matrixf currentSample = {
                    1, inputSize, &trainingData->data[sample * inputSize]
            };
            // update for kernel
            applyGaussianFilter(&currentSample, &Kernel);
            // Forward propagation
            forwardPropagation(nn, currentSample.data, hiddenLayer, outputLayer);

            // Calculate loss (e.g., mean squared error)
            float loss = 0.0f;
                for (int i = 0; i < outputSize; i++) {
                    float target = (i == (int)labels->data[sample]) ? 1.0f : 0.0f;  // Assuming labels are one-hot encoded
                    loss += 0.5f * (outputLayer[i] - target) * (outputLayer[i] - target);
            }

            // Accumulate total loss
            totalLoss += loss;
            backwardPropagation(nn, &currentSample, hiddenLayer, outputLayer, labels->data[sample], learningRate);
        }
        
        printf("Epoch %d/%d, Loss: %f\n", epoch + 1, epochs, totalLoss / numSample);
    }
}

void predict(NeuralNetwork *nn, Matrixf *inputData) {
    float hiddenLayer[nn->hiddenSize];
    float outputLayer[nn->outputSize];
    
    // Forward propagation untuk setiap baris dalam inputData
    for (int i = 0; i < inputData->rows; i++) {
        // Ambil satu baris dari inputData
        float *inputRow = &inputData->data[i * inputData->cols];
        
        // Lakukan forward propagation
        forwardPropagation(nn, inputRow, hiddenLayer, outputLayer);

        // Tampilkan hasil prediksi
        int predicted = (outputLayer[0] > 0.5f) ? 1 : 0;
        printf("Prediction for predicted %d\n",predicted);
    }
}



void freeNeuralNetwork(NeuralNetwork* nn){
    // clear input bias
    if(nn->hiddenBias){
        nn->hiddenBias = NULL;
        nn->inputSize = NULL;
    }
}
#endif 