#include <stdio.h>
#include <stdlib.h>
#include "utils.h"
#include "NN.h"

int main(void) {
    Matrix mat;
    mat.rows = 4;
    mat.cols = 2;

    float tempCelciusFahrenheit[4][2] = {
        {0.0, 32.0},   // 0°C = 32°F
        {100.0, 212.0}, // 100°C = 212°F
        {20.0, 68.0},  // 20°C = 68°F
        {30.0, 86.0}   // 30°C = 86°F
    };

    initialize_matrix(&mat, (float*)tempCelciusFahrenheit);
    print_matrix(&mat);

    float *expected_output = (float *)malloc(mat.rows * sizeof(float));

    for (int i = 0; i < mat.rows; i++) {
        expected_output[i] = tempCelciusFahrenheit[i][1];
    }

    Perceptron p;
    initialize_perceptron(&p, 2); 
    train_perceptron_regression(&p, &mat, expected_output, 1000, 0.00001f);  // Train perceptron

    printf("Predicted Output:\n");
    for (int i = 0; i < mat.rows; i++) {
        float output = predict_regression(&p, tempCelciusFahrenheit[i]);
        printf("Input: %.1f, %.1f - Predicted Output: %.1f\n", tempCelciusFahrenheit[i][0], tempCelciusFahrenheit[i][1], output);
    }

    free(expected_output);
    return 0;
}
