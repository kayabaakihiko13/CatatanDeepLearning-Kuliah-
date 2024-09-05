#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "NN.h"

int main(void) {
    // Inisialisasi matriks untuk input data
    Matrix mat;
    mat.rows = 4;
    mat.cols = 2;

    // Data input: konversi suhu Celcius ke Fahrenheit
    float tempCelciusFahrenheit[5][2] = {
        {0.0, 32.0},    // 0°C = 32°F
        {100.0, 212.0}, // 100°C = 212°F
        {20.0, 68.0},   // 20°C = 68°F
        {30.0, 86.0},  // 30°C = 86°F
        {60.0, 117.0},  // 30°C = 86°F
    };

    // Inisialisasi matriks
    initialize_matrix(&mat, (float *)tempCelciusFahrenheit);
    print_matrix(&mat); // Cetak matriks

    // Alokasi memori untuk output yang diharapkan
    float *expected_output = (float *)malloc(mat.rows * sizeof(float));

    // Isi output yang diharapkan dengan nilai Fahrenheit
    for (int i = 0; i < mat.rows; i++) {
        expected_output[i] = tempCelciusFahrenheit[i][1];
    }

    // Inisialisasi objek SingleLayerPerceptronRegression
    SingleLayerPerceptronRegression perceptron;
    Perceptron p;
    perceptron.initialize_perceptron(&p, 2); 

    // Latih perceptron dengan data input dan output yang diharapkan
    perceptron.train_perceptron_regression(&p, &mat, expected_output, 1000, 0.00006f);

    // Prediksi output menggunakan perceptron yang telah dilatih
    printf("Predicted Output:\n");
    for (int i = 0; i < mat.rows; i++) {
        float output = perceptron.predict_regression(&p, tempCelciusFahrenheit[i]);
        printf("Input: %.1f, %.1f - Predicted Output: %.1f\n", tempCelciusFahrenheit[i][0], tempCelciusFahrenheit[i][1], output);
    }

    // Bebaskan memori
    free(expected_output);
    return 0;
}
