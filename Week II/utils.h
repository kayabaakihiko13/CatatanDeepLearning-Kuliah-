#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include "utils.h"

typedef struct {
    // aturan tipe data format Matrix
    int rows;
    int cols;
    float **data;
} Matrix;
// Inisialisasi matriks
/*
    * inisialisasi ini matriks ini dimana mengubah data array 2 secara mentah kedalam 
    * format yang sesuai diatur 
*/
void initialize_matrix(Matrix *mat, const float *data) {
    // membuat alokasi memory pada matrix untuk banyaknya baris
    mat->data = (float **)malloc(mat->rows * sizeof(float *));
    for (int i = 0; i < mat->rows; i++) {
        // lalu kita membuat alokasi kolom pada setiap baris
        mat->data[i] = (float *)malloc(mat->cols * sizeof(float));
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = *(data + i * mat->cols + j);
        }
    }
}

// Bebaskan memori matriks
void free_matrix(Matrix *mat) {
    // kita melepaskan memori addreas pada matriks
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
}

// Cetak matriks
void print_matrix(const Matrix *mat) {
    // output untuk matrix
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            printf("%.1f ", mat->data[i][j]);
        }
        printf("\n");
    }
}

#endif