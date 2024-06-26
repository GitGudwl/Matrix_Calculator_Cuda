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

void matrixMultiplicationCuda(double* matrix1, double* matrix2, double* result, int size, int block) {
    cout << "multlipying Please Wait\n";
    float time;

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    cudaError_t err;
    cudaEvent_t start, stop;

    // Create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix1, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix2, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, size * size * sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((size + block - 1) / block, (size + block - 1) / block);

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch the kernel
    matrixMultiplication << <numBlocks, threadsPerBlock >> > (d_matrix1, d_matrix2, d_result, size);

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "Time to Multiply 2 matrices: " << time << " ms\n";

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, size * size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix1));
    CHECK_CUDA_ERROR(cudaFree(d_matrix2));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    // print the result
    savetofile(result, "multiplication.csv", size, size);
    //print the result
}

void matrixSubtractionCuda(double* matrix1, double* matrix2, double* result, int size, int block) {
    cout << "Substracting Please Wait\n";
    float time;

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    cudaError_t err;
    cudaEvent_t start, stop;

    // Create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix1, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix2, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, size * size * sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((size + block - 1) / block, (size + block - 1) / block);

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch the kernel
    matrixSubstraction << <numBlocks, threadsPerBlock >> > (d_matrix1, d_matrix2, d_result, size);

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "Time to substract matrices: " << time << " ms\n";

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, size * size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix1));
    CHECK_CUDA_ERROR(cudaFree(d_matrix2));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    // print the result
    savetofile(result, "Substract.csv", size, size);
    //print the result

}

void matrixDivisionCuda(double* matrix1, double* matrix2, double* result, int size, int block) {
    std::cout << "Divide, please wait...\n";
    float time;

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix1, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix2, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, size * size * sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((size + block - 1) / block, (size + block - 1) / block);

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch the kernel
    matrixDivision << <numBlocks, threadsPerBlock >> > (d_matrix1, d_matrix2, d_result, size);

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    std::cout << "Time to divide matrices: " << time << " ms\n";

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, size * size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix1));
    CHECK_CUDA_ERROR(cudaFree(d_matrix2));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

    // Save the result to a file
    savetofile(result, "Divide.csv", size, size);
}

void matrixAdditionCuda(double* matrix1, double* matrix2, double* result, int size, int block) {
    cout << "Adding Please Wait\n";
    float time;

    double* d_matrix1;
    double* d_matrix2;
    double* d_result;

    cudaError_t err;
    cudaEvent_t start, stop;

    // Create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix1, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix2, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, size * size * sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix1, matrix1, size * size * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix2, matrix2, size * size * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((size + block - 1) / block, (size + block - 1) / block);

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch the kernel
    matrixAddition << <numBlocks, threadsPerBlock >> > (d_matrix1, d_matrix2, d_result, size);

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "Time to add matrices: " << time << " ms\n";

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, size * size * sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix1));
    CHECK_CUDA_ERROR(cudaFree(d_matrix2));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
	// print the result
	savetofile(result, "add.csv", size, size);
	//print the result

}

void matrixDeterminantCuda(double* matrix, double* result, int size, int block) {
    cout << "Calculating Determinant, Please Wait\n";
    float time;

    double* d_matrix;
    int* d_pivots;
    int* d_error;
    double* d_result;

    cudaError_t err;
    cudaEvent_t start, stop;

    // Create CUDA events for timing
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_matrix, size * size * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_pivots, size * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_error, sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_result, sizeof(double)));

    // Copy data from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_matrix, matrix, size * size * sizeof(double), cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 threadsPerBlock(block, block);
    dim3 numBlocks((size + block - 1) / block, 1);

    // Record the start event
    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));

    // Launch the LU Decomposition kernel
    luDecomposition << <numBlocks, threadsPerBlock >> > (d_matrix, size, d_pivots, d_error);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Check for LU decomposition errors
    int error;
    CHECK_CUDA_ERROR(cudaMemcpy(&error, d_error, sizeof(int), cudaMemcpyDeviceToHost));
    if (error)
    {
        std::cerr << "Error: Singular matrix encountered." << std::endl;
        exit(EXIT_FAILURE);
    }

    // Launch determinant calculation kernel
    determinant << <1, 1 >> > (d_matrix, size, d_pivots, d_result);
    CHECK_CUDA_ERROR(cudaGetLastError());

    // Record the stop event
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    // Calculate the elapsed time
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&time, start, stop));
    cout << "Time to calculate determinant: " << time << " ms\n";

    // Copy the result back to the host
    CHECK_CUDA_ERROR(cudaMemcpy(result, d_result, sizeof(double), cudaMemcpyDeviceToHost));

    // Free device memory
    CHECK_CUDA_ERROR(cudaFree(d_matrix));
    CHECK_CUDA_ERROR(cudaFree(d_pivots));
    CHECK_CUDA_ERROR(cudaFree(d_error));
    CHECK_CUDA_ERROR(cudaFree(d_result));

    // Destroy CUDA events
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));

	cout << "Determinant: " << *result << std::endl;
}

void matrixInverseCUDA(double* L, double* iL, int n, int block) {
    cout << "Inversing Please Wait\n";

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
    savetofile(iL, "inv.csv", n, n);

    cudaFree(d_A);
    cudaFree(dI);

    delete[] I;
}

#endif // MATRIX_BASIC_CALCULATOR