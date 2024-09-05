#include <stdio.h>
#include <stdlib.h> // Untuk malloc dan free
#include "NN.h"
#include "utils.h"

// Fungsi untuk menulis data ke file CSV
void write_to_csv(const char *filename, const Matrix *inputs, const float *outputs, const int *predictions) {
    FILE *file = fopen(filename, "w");
    if (file == NULL) {
        fprintf(stderr, "Error: Tidak dapat membuka file untuk ditulis.\n");
        return;
    }

    // Tulis header ke file CSV
    fprintf(file, "Input1,Input2,Output,Expected\n");

    // Loop untuk menulis setiap baris data ke file CSV
    for (int i = 0; i < inputs->rows; i++) {
        fprintf(file, "%.0f,%.0f,%d,%.0f\n", 
                inputs->data[i][0], inputs->data[i][1], predictions[i], outputs[i]);
    }

    fclose(file);
}

int main() {
    // Inisialisasi data XOR
    Matrix NORDATA;
    NORDATA.rows = 4;
    NORDATA.cols = 2;
    float nor_inputs[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    initialize_matrix(&NORDATA, (float *)nor_inputs);
    float nor_outputs[4] = {0, 1, 1, 0};
    // Inisialisasi perceptron
    Perceptron nor_perceptron;
    initialize_perceptron(&nor_perceptron, NORDATA.cols);

    // Melatih perceptron
    int epochs = 1000;
    float learning_rate = 0.009;
    trainPerceptornBackprogration(&nor_perceptron, &NORDATA, nor_outputs, epochs, learning_rate);

    // Array untuk menyimpan prediksi
    int predictions[NORDATA.rows];

    // Menguji perceptron dan menyimpan hasil prediksi
    printf("Hasil prediksi setelah pelatihan:\n");
    for (int i = 0; i < NORDATA.rows; i++) {
        float prediction = predict_classification(&nor_perceptron, NORDATA.data[i]);
        predictions[i] = (prediction >= 0.5f) ? 1 : 0; // Mengubah float ke kelas biner 0 atau 1
        printf("Input: %.0f %.0f, Output: %d, Expected: %.0f\n", 
               NORDATA.data[i][0], NORDATA.data[i][1], predictions[i], nor_outputs[i]);
    }

    // Tulis hasil prediksi ke file CSV
    write_to_csv("results_nor.csv", &NORDATA, nor_outputs, predictions);

    // Membersihkan memori
    for (int i = 0; i < NORDATA.rows; i++) {
        free(NORDATA.data[i]);
    }
    free(NORDATA.data);
    free(nor_perceptron.weights);

    return 0;
}
