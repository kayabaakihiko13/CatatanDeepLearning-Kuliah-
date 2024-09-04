#ifndef NN_H
#define NN_H

#include <stdlib.h>
#include <math.h>
#include "NN.h"
#include "utils.h"
// Fungsi sigmoid dan turunannya
typedef struct {
    // tipe data untuk Perceptron tersebut;
    int input_size;
    float *weights;
    float bias;
} Perceptron;
static float sigmoid(float x) {
    // ini merupakan code untuk kalkulasi sigmoid
    return 1.0 / (1.0 + exp(-x));
}

static float sigmoid_derivative(float x) {
    // ini merupakan untuk turunan dari sigmoid
    return x * (1.0 - x);
}

// Inisialisasi perceptron
void initialize_perceptron(Perceptron *p, int input_size) {
    p->input_size = input_size;
    p->weights = (float *)malloc(input_size * sizeof(float));
    for (int i = 0; i < input_size; i++) {
        p->weights[i] = ((float)rand() / RAND_MAX) * 0.01f;
    }
    p->bias = ((float)rand() / RAND_MAX) * 0.01f;
}

// Latih perceptron
void train_perceptron_regression(Perceptron *p, const Matrix *inputs, const float *expected_output, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int i = 0; i < inputs->rows; i++) {
            float weighted_sum = 0.0;
            for (int j = 0; j < p->input_size; j++) {
                weighted_sum += p->weights[j] * inputs->data[i][j];
            }
            weighted_sum += p->bias;

            float output = weighted_sum; // For regression, we don't use an activation function
            float error = expected_output[i] - output;

            for (int j = 0; j < p->input_size; j++) {
                p->weights[j] += learning_rate * error * inputs->data[i][j];
            }
            p->bias += learning_rate * error;
        }
    }
}

void train_perceptron_classification(Perceptron *p,const Matrix *inputs,const float *expected_output,int epochs,float learning_rate){
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
                p->weights[j] += learning_rate * error * sigmoid_derivative(output) * inputs->data[i][j];
            }
            p->bias += learning_rate * error *sigmoid_derivative(output);
        }
    }
}

// Prediksi output dengan perceptron
float predict_classification(const Perceptron *p, const float input[]) {
    // fungsi ini dimana untuk hasil dari bobot yang telah ditrain
    // lalu kita ubah kedalam sigmoid
    float weighted_sum = 0.0;
    for (int i = 0; i < p->input_size; i++) {
        weighted_sum += p->weights[i] * input[i];
    }
    weighted_sum += p->bias;
    return sigmoid(weighted_sum);
}
float predict_regression(const Perceptron *p, const float input[]) {
    float weighted_sum = 0.0;
    for (int i = 0; i < p->input_size; i++) {
        weighted_sum += p->weights[i] * input[i];
    }
    weighted_sum += p->bias;
    // For regression, we return the raw weighted sum
    return weighted_sum; 
}
#endif