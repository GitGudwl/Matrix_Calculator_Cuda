// matrix_inversion.cuh

#ifndef MATRIX_INVERSION_CUH
#define MATRIX_INVERSION_CUH
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "utility.h"
using namespace std;
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << " (" << err << ")" << std::endl; \
        __debugbreak(); \
    } \
} while (0)

// Normalize non-diagonal elements
__global__ void nodiag_normalize(double* A, double* I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == i && x != y)
		{
			I[x * n + y] /= A[i * n + i];
			A[x * n + y] /= A[i * n + i];
		}
}

// Normalize diagonal elements
__global__ void diag_normalize(double* A, double* I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
		if (x == y && x == i)
		{
			I[x * n + y] /= A[i * n + i];
			A[x * n + y] /= A[i * n + i];
		}
}

// Apply Gauss-Jordan elimination
__global__ void gaussjordan(double* A, double* I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	{
		if (x != i)
		{
			I[x * n + y] -= I[i * n + y] * A[x * n + i];
			if (y != i)
			{
				A[x * n + y] -= A[i * n + y] * A[x * n + i];
			}
		}
	}
}

// Set non-diagonal elements to zero
__global__ void set_zero(double* A, double* I, int n, int i)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < n && y < n)
	{
		if (x != i)
		{
			if (y == i)
			{
				A[x * n + y] = 0;
			}
		}
	}
}

//addition of two matrices
__global__ void matrixAddition(float* a, float* b, float* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int index = row * n + col;
        c[index] = a[index] + b[index];
    }
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

#endif // MATRIX_INVERSION_CUH