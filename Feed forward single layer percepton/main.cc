#include <stdio.h>
#include "utils.h"
#include "NN.h"

int main(void) {
    // Create a matrix for the AND Gate
    Matrix mat;
    mat.rows = 4;
    mat.cols = 2;

    // Initialize matrix with AND Gate data
    float data[4][2] = {
        {0.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 0.0f},
        {1.0f, 1.0f}
    };

    initialize_matrix(&mat, (float *)data);
    print_matrix(&mat);

    // Neural network operation
     // Expected output for AND Gate
    float expected_output[4] = {0.0f, 0.0f, 0.0f, 1.0f}; 
    Perceptron p;
     // Initialize perceptron with 2 inputs
    initialize_perceptron(&p, 2); 
    train_perceptron_classification(&p, &mat, expected_output, 1000, 0.1f);  // Train perceptron

    // Print trained weights and bias
    printf("Trained weights: [%.5f, %.5f]\n", p.weights[0], p.weights[1]);
    printf("Trained bias: %.5f\n", p.bias);

    // Prediction with input data
    for (int i = 0; i < mat.rows; i++) {
        float input[2] = {mat.data[i][0], mat.data[i][1]};
        float output = predict_classification(&p, input);
        // Classification threshold
        int classified_output = (output >= 0.5f) ? 1 : 0;  
        printf("Input: %.0f, %.0f - Predicted Output: %.5f - Classified Output: %d\n", 
               input[0], input[1], output, classified_output);
    }

    // Free matrix memory
    free_matrix(&mat);

    return 0;
}
