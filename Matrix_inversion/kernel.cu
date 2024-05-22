
#include "Header.cuh"
#include "utility.h"
using namespace std;

#define blocksize 8

int main()
{
	int n;
	string filename;

	cout << "Enter the matrix size: ";
	cin >> n;

	double* result = new double[n * n];
	double* L = new double[n * n];

	cout << "Enter the filename: ";	
	cin >> filename;

	matrix_read(L, n, filename.c_str());

	matrixInverseCUDA(L, result, n, blocksize);
	
	delete[] L;
	delete[] result;
	
	system("Pause");
	return 0;
}

