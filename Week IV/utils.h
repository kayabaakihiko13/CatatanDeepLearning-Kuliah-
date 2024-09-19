#ifndef UTILS_H
#define UTILS_H


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <random>

/**
    * this for matrix with float datatype
 */
typedef struct {
    int rows;
    int cols;
    float *data;
} Matrixf;
void initializationMatrixf(Matrixf *mat,int rows,int cols){
    // initialization matrix
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (float*)malloc(rows*cols*sizeof(float));
}

Matrixf randomMatrixf(int rows, int cols){
    Matrixf mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = (float*)malloc(rows * cols * sizeof(float));
    for(int i = 0; i < mat.rows; i++){
        for(int j = 0; j < mat.cols; j++){
            mat.data[i * cols + j] = (float)rand() / RAND_MAX;
        }
    }
    return mat;
}
void ConvertArrayToMatrix(Matrixf *mat, float* arr, const int rows, const int cols){
    if((rows * cols) != (rows * cols)){
        fprintf(stderr, "size in matrix and array not match");
    }
    for(int i = 0; i < mat->rows; i++){
        for(int j = 0; j < mat->cols; j++){
            mat->data[i * cols + j] = arr[i * cols + j];
        }
    }
}
Matrixf MatmulNaive(Matrixf *A, Matrixf *B) {
    if (A->cols != B->rows) {
        fprintf(stderr, "Error: Matrix dimensions do not match for multiplication.\n");
        exit(1);
    }

    Matrixf C;
    C.rows = A->rows;
    C.cols = B->cols;
    C.data = (float*)malloc(C.rows * C.cols * sizeof(float));

    for (int i = 0; i < C.rows; i++) {
        for (int j = 0; j < C.cols; j++) {
            C.data[i * C.cols + j] = 0;
            for (int k = 0; k < A->cols; k++) {
                C.data[i * C.cols + j] += A->data[i * A->cols + k] * B->data[k * B->cols + j];
            }
        }
    }

    return C;
}
void TransformMatrixf(Matrixf *mat) {
    if (mat->data == NULL) {
        fprintf(stderr, "Your data is empty\n");
        return;
    }

    Matrixf transposed;
    transposed.rows = mat->cols;
    transposed.cols = mat->rows;
    transposed.data = (float*)malloc(transposed.rows * transposed.cols * sizeof(float));

    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            transposed.data[j * transposed.cols + i] = mat->data[i * mat->cols + j];
        }
    }

    // Replace original matrix with transposed matrix
    free(mat->data);
    mat->data = transposed.data;
    mat->rows = transposed.rows;
    mat->cols = transposed.cols;
}
void ReadMatrixf(Matrixf *mat,int size){
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            printf("%f",mat->data[i*mat->cols+j]);
        }
        printf("\n");
    }
    printf("\n");
}

typedef struct{
    int rows;
    int cols;
    int *data;
} Matrixi;



#endif