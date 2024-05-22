#ifndef MATRIX_BASIC_CALCULATOR
#define MATRIX_BASIC_CALCULATOR

#include "utility.h"
#include "Header.cuh"

#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        __debugbreak(); \
    } \
} while (0)


void matrixAddition(double* matrix1, double* matrix2, double* result, int size) {
   
}

void matrixMultiplication(double* matrix1, double* matrix2, double* result, int size) {
    // Implementation of matrix multiplication
}

void matrixSubtraction(double* matrix1, double* matrix2, double* result, int size) {
    // Implementation of matrix subtraction
}

void matrixDivision(double* matrix1, double* matrix2, double* result, int size) {
    // Implementation of matrix division
}


void matrixInverseCUDA(double* L, double* iL, int n, int block) {
    cout << "inv\n";

    double* d_A;
    double* dI;
    double* I;
    float time;
    cudaError_t err;
    cudaEvent_t start, stop;

    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    int ddsize = n * n * sizeof(double);

    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((n + block - 1) / block, (n + block - 1) / block);

    // Memory allocation
    err = cudaMalloc((void**)&d_A, ddsize);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMalloc((void**)&dI, ddsize);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    I = new double[n * n];

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                I[i * n + i] = 1.0;
            else
                I[i * n + j] = 0.0;
        }
    }

    // Copy data from CPU to GPU
    err = cudaMemcpy(d_A, L, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(dI, I, ddsize, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    // Timer start
    cudaEventRecord(start, 0);

    // L^(-1)
    for (int i = 0; i < n; i++) {
        nodiag_normalize << <numBlocks, threadsPerBlock >> > (d_A, dI, n, i);
        diag_normalize << <numBlocks, threadsPerBlock >> > (d_A, dI, n, i);
        gaussjordan << <numBlocks, threadsPerBlock >> > (d_A, dI, n, i);
        set_zero << <numBlocks, threadsPerBlock >> > (d_A, dI, n, i);
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy data from GPU to CPU
    err = cudaMemcpy(iL, dI, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }
    err = cudaMemcpy(I, d_A, ddsize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    cout << "Cuda Time - inverse: " << time << "ms\n";
    savetofile(iL, "inv.txt", n, n);
    savetofile(I, "I.txt", n, n);

    cudaFree(d_A);
    cudaFree(dI);

    delete[] I;
}

#endif // MATRIX_BASIC_CALCULATOR