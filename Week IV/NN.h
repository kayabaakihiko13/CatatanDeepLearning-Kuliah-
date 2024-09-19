#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "utils.h"
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
inline float randomFloat(){
    return (float)rand() / (float)RAND_MAX * 2 - 1;
};
Matrixf kernelConvolution(int* size) {
    int rowSize = size[0];
    int colSize = size[1];
    
    Matrixf kernel;
    kernel.rows = rowSize;
    kernel.cols = colSize;
    kernel.data = (float*)malloc(rowSize * colSize * sizeof(float));
    
    if (kernel.data == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    
    // Inisialisasi generator angka acak
    srand(time(NULL));
    
    // Isi kernel dengan nilai acak
    for (int i = 0; i < rowSize * colSize; i++) {
        kernel.data[i] = randomFloat();
    }
    
    // Normalisasi kernel
    float sum = 0.0f;
    for (int i = 0; i < rowSize * colSize; i++) {
        sum += kernel.data[i];
    }
    
    if (sum != 0) {
        for (int i = 0; i < rowSize * colSize; i++) {
            kernel.data[i] /= sum;
        }
    }
    
    return kernel;
}



#endif 