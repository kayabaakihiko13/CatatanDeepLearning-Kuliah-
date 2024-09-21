#include <stdio.h>
#include "utils.h"
#include "NN.h"

int main() {
    Matrixf InputMat;
    Matrixf TargetMat;
    float inputRecord[8] = {0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0};
    float targetRecord[4] = {0.0, 1.0, 1.0, 0.0};

    // Initialize matrices
    initializationMatrixf(&InputMat, 4, 2);
    initializationMatrixf(&TargetMat, 4, 1);
    // convert dulu bozz!!
    ConvertArrayToMatrix(&InputMat, inputRecord, 4, 2);
    ConvertArrayToMatrix(&TargetMat, targetRecord, 4, 1);
    
    printf("size of Input record is %dX%d",InputMat.rows,InputMat.cols);
    printf("size of Target record is %dX%d",TargetMat.rows,TargetMat.cols);
    NeuralNetwork nn = initializationNetwork(InputMat.rows*InputMat.cols, 100, 2);
    trainNetwork(&nn, &InputMat, &TargetMat, InputMat.rows, 100, 0.00001f);
    // float result[4];
    predict(&nn, &InputMat);

    return 0;
}