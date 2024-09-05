#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>
#include "NN.h"
#include "utils.h"

// Struktur untuk Perceptron
typedef struct {
    int input_size;
    float *weights;
    float bias;
} Perceptron;

// Kelas untuk Single Layer Perceptron untuk Regresi
class SingleLayerPerceptronRegression {
private:


public:
    // Inisialisasi perceptron
    void initialize_perceptron(Perceptron *p, int input_size) {
        p->input_size = input_size;
        p->weights = (float *)malloc(input_size * sizeof(float));
        for (int i = 0; i < input_size; i++) {
            p->weights[i] = ((float)rand() / RAND_MAX) * 0.01f;
        }
        p->bias = ((float)rand() / RAND_MAX) * 0.01f;
    }
    // Latih perceptron untuk regresi
    void train_perceptron_regression(Perceptron *p, const Matrix *inputs, const float *expected_output, int epochs, float learning_rate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs->rows; i++) {
                float weighted_sum = 0.0;
                for (int j = 0; j < p->input_size; j++) {
                    weighted_sum += p->weights[j] * inputs->data[i][j];
                }
                weighted_sum += p->bias;

                float output = weighted_sum; // Tidak ada fungsi aktivasi untuk regresi
                float error = expected_output[i] - output;

                for (int j = 0; j < p->input_size; j++) {
                    p->weights[j] += learning_rate * error * inputs->data[i][j];
                }
                p->bias += learning_rate * error;
            }
        }
    }

    // Prediksi output untuk regresi
    float predict_regression(const Perceptron *p, const float input[]) {
        float weighted_sum = 0.0;
        for (int i = 0; i < p->input_size; i++) {
            weighted_sum += p->weights[i] * input[i];
        }
        weighted_sum += p->bias;
        return weighted_sum; // Mengembalikan hasil weighted sum
    }
};

// Kelas untuk Single Layer Perceptron untuk Klasifikasi
class SingleLayerPerceptronClassification {
private:
    // Fungsi sigmoid
    float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    // Turunan dari fungsi sigmoid
    float sigmoidDerivative(float x) {
        return x * (1.0 - x);
    }

public:
    // Inisialisasi perceptron
    void initialize_perceptron(Perceptron *p, int input_size) {
        p->input_size = input_size;
        p->weights = (float *)malloc(input_size * sizeof(float));
        for (int i = 0; i < input_size; i++) {
            p->weights[i] = ((float)rand() / RAND_MAX) * 0.01f;
        }
        p->bias = ((float)rand() / RAND_MAX) * 0.01f;
    }
    // Latih perceptron untuk klasifikasi
    void train_perceptron_classification(Perceptron *p, const Matrix *inputs, const float *expected_output, int epochs, float learning_rate) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int i = 0; i < inputs->rows; i++) {
                float weighted_sum = 0.0;
                for (int j = 0; j < p->input_size; j++) {
                    weighted_sum += p->weights[j] * inputs->data[i][j];
                }
                weighted_sum += p->bias;

                float output = sigmoid(weighted_sum);
                float error = expected_output[i] - output;

                for (int j = 0; j < p->input_size; j++) {
                    p->weights[j] += learning_rate * error * sigmoidDerivative(output) * inputs->data[i][j];
                }
                p->bias += learning_rate * error * sigmoidDerivative(output);
            }
        }
    }

    // Prediksi output untuk klasifikasi
    float predict_classification(const Perceptron *p, const float input[]) {
        float weighted_sum = 0.0;
        for (int i = 0; i < p->input_size; i++) {
            weighted_sum += p->weights[i] * input[i];
        }
        weighted_sum += p->bias;
        return sigmoid(weighted_sum);
    }
};

// Single Layer Perceptron with Backprogration

inline float Sigmoid(float x){
    return 1.0/(1.0 + exp(-x));
}
inline float sigmoidDerivative(float x) {
        return x * (1.0 - x);
    }
void initialize_perceptron(Perceptron *perceptron,int inputSize){
    perceptron->input_size = inputSize;
    perceptron->weights = (float *)malloc(inputSize * sizeof(float));
        for (int i = 0; i < inputSize; i++) {
            perceptron->weights[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
    perceptron->bias = ((float)rand() / RAND_MAX) *2 - 1;
}

void trainPerceptornBackprogration(Perceptron *perceptron,const Matrix *inputs,const float*expectedOutput,int epochs,float learningRate){
    for(int epoch=0;epoch<epochs;epoch++){
        for(int i = 0;i<inputs->rows;i++){
            float weightSum = 0.0;
            for(int j= 0;j<inputs->rows;j++){
                weightSum += perceptron->weights[j] * inputs->data[i][j];     
            }
            weightSum += perceptron->bias;
            float output  = Sigmoid(weightSum);
            // backprogration
            float error = expectedOutput[i] -output;
            float delta = error*output * (1-output);
            // update bobot dan bias nya
            for(int j = 0;j<perceptron->input_size;j++){
                perceptron->weights[j] += learningRate * delta * inputs->data[i][j];
            }
            perceptron->bias +=learningRate * delta;
        }
    }
    
}
float predict_classification(const Perceptron *p, const float input[]) {
        float weighted_sum = 0.0;
        for (int i = 0; i < p->input_size; i++) {
            weighted_sum += p->weights[i] * input[i];
        }
        weighted_sum += p->bias;
        return Sigmoid(weighted_sum);
}

#endif // NN_H
