#include "utility.h"
#include "MatrixCalculator.cuh"
using namespace std;

#define blocksize 8

#include <iostream>
#include <string>
using namespace std;

int main() {
    int n;
    string filename;
    string filename2;

    cout << "Enter the matrix size: ";
    cin >> n;

    double* result = new double[n * n];
    double* L = new double[n * n];
    double* M = new double[n * n];

    cout << "Enter the filename :\n (content of the file suppose to be csv or txt seperated by comma)";
    cin >> filename;

    matrix_read(L, n, filename.c_str());

    int operation;
    cout << "\nSelect operation:\n";
    cout << "1. Addition\n";
    cout << "2. Multiplication\n";
    cout << "3. Subtraction\n";
    cout << "4. Division\n";
	cout << "5. Inversion\n";
	cout << "Enter operation: ";
    cin >> operation;

    cout << "\nEnter the second filename: ";
    cout << "\n!Make sure that the second matrix is square matrix with same dimension\n\n\n";

    switch (operation) {
    case 1:
		cin >> filename2;
        matrix_read(M, n, filename2.c_str());
        matrixAdditionCuda(L, M, result, n,blocksize);
        break;
    case 2:
        cin >> filename2;
        matrix_read(M, n, filename2.c_str());
        matrixMultiplicationCuda(L, M, result, n,blocksize);
        break;
    case 3:
        cin >> filename2;
        matrix_read(M, n, filename2.c_str());
        matrixSubtractionCuda(L, M, result, n,blocksize);
        break;
    case 4:
        cin >> filename2;
        matrix_read(M, n, filename2.c_str());
        matrixDivisionCuda(L, M, result, n,blocksize);
        break;
	case 5:
        matrixInverseCUDA(L, result, n, blocksize);
		break;
    default:
        cout << "Invalid operation\n";
        break;
    }

    delete[] L;
    delete[] result;
    delete[] M;

    system("Pause");
    return 0;
}


