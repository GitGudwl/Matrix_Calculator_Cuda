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
__global__ void matrixAddition(double* a, double* b, double* c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int index = row * n + col;
        c[index] = a[index] + b[index];
    }
}

//Substract of two matrices
__global__ void matrixSubstraction(double* a, double* b, double* c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < n) {
		int index = row * n + col;
		c[index] = a[index] - b[index];
	}
}

//Multiplication of two matrices
__global__ void matrixMultiplication(double* a, double* b, double* c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < n) {
		float sum = 0;
		for (int i = 0; i < n; i++) {
			sum += a[row * n + i] * b[i * n + col];
		}
		c[row * n + col] = sum;
	}
}

//Division of two matrices
__global__ void matrixDivision(double* a, double* b, double* c, int n) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < n && col < n) {
		c[row * n + col] = a[row * n + col] / b[row * n + col];
	}
}
// LU Decomposition kernel
__global__ void luDecomposition(double* A, int n, int* pivots, int* error)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < n)
	{
		pivots[tid] = tid;
	}

	for (int i = 0; i < n; ++i)
	{
		if (tid < i)
		{
			continue;
		}

		if (A[i * n + i] == 0)
		{
			*error = 1; // Set error flag
			return;
		}

		for (int j = i + 1; j < n; ++j)
		{
			if (tid < j)
			{
				A[j * n + i] /= A[i * n + i];
			}

			__syncthreads();

			if (tid < j)
			{
				for (int k = i + 1; k < n; ++k)
				{
					A[j * n + k] -= A[j * n + i] * A[i * n + k];
				}
			}
		}
	}
}

// Determinant calculation kernel
__global__ void determinant(double* A, int n, int* pivots, double* result)
{
	if (blockIdx.x == 0 && threadIdx.x == 0)
	{
		*result = 1.0;

		for (int i = 0; i < n; ++i)
		{
			*result *= A[i * n + i];
		}

		// Account for row swaps
		for (int i = 0; i < n; ++i)
		{
			if (pivots[i] != i)
			{
				*result *= -1.0;
			}
		}
	}
}



#endif // MATRIX_INVERSION_CUH