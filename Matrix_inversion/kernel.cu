#include "utility.h"
#include "MatrixCalculator.cuh"
#include <iostream>
#include <string>
using namespace std;


#define blocksize 8


int main() {
    int n;
    string filename;
    string filename2;

    cout << "Welcome to the Matrix Calculator!" << endl;

    // Input matrix size
    cout << "Enter the size of the square matrix (e.g., 3 for a 3x3 matrix): ";
    while (!(cin >> n) || n <= 0) {
        cout << "Invalid input. Please enter a positive integer for the matrix size: ";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }


    double* result = new double[n * n];
    double* L = new double[n * n];
    double* M = new double[n * n];

    // Input first matrix filename
    cout << "Please enter the filename of the first matrix (CSV or TXT format, separated by commas): ";
    cin >> filename;
    matrix_read(L, n, filename.c_str());

    // Display menu and select operation
    int operation;
    cout << "\nSelect an operation to perform:\n";
    cout << "1. Addition\n";
    cout << "2. Multiplication\n";
    cout << "3. Subtraction\n";
    cout << "4. Determinant\n";
    cout << "5. Inversion\n";
    cout << "Enter your choice (1-5): ";
    while (!(cin >> operation) || operation < 1 || operation > 5) {
        cout << "Invalid choice. Please enter a number between 1 and 5: ";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    // Input second matrix filename if necessary
    if (operation <4) {
        cout << "\nPlease enter the filename of the second matrix (CSV or TXT format, separated by commas): ";
        cout << "\nMake sure that the second matrix is a square matrix with the same dimensions.\n";
        cin >> filename2;
        matrix_read(M, n, filename2.c_str());
    }

    // Perform the selected operation
    switch (operation) {
    case 1:
        matrixAdditionCuda(L, M, result, n, blocksize);
        cout << "Matrix addition completed.\n";
        break;
    case 2:
        matrixMultiplicationCuda(L, M, result, n, blocksize);
        cout << "Matrix multiplication completed.\n";
        break;
    case 3:
        matrixSubtractionCuda(L, M, result, n, blocksize);
        cout << "Matrix subtraction completed.\n";
        break;
    case 4:
        matrixDeterminantCuda(L,result, n, blocksize);
        cout << "Matrix Determinant Calculation completed.\n";
        break;
    case 5:
        matrixInverseCUDA(L, result, n, blocksize);
        cout << "Matrix inversion completed.\n";
        break;
    default:
        cout << "Invalid operation selected.\n";
        break;
    }

    // Clean up
    delete[] L;
    delete[] result;
    delete[] M;

    cout << "Operation completed. Press Enter to exit.";
    cin.ignore();
    cin.get();
    return 0;
}
