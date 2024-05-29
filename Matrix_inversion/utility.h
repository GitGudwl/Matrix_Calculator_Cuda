#pragma once
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;
void savetofile(double* A, string s, int n, int h)
{
	std::ofstream plik;
	plik.open(s);

	for (int j = 0; j < h; j++)
	{
		for (int i = 0; i < h; i++)
		{
			plik << A[j * n + i] << ",";
		}
		plik << endl;
	}
	plik.close();
}


#include <stdio.h>
#include <stdlib.h>

int matrix_read(double* L, int dimension, const char* filename)
{
	//check if the file extension is .csv or .txt 
	string s = filename;
    if (s.find(".csv") == string::npos && s.find(".txt") == string::npos) {
		perror("Error: File extension must be .csv or .txt");
		system("pause");
		exit(EXIT_FAILURE);
    }
    FILE* fp;
    int row = 0, col = 0;
    int actual_rows = 0, actual_cols = 0;
    int temp_col_count = 0;
    double temp;

    // Open the file
    fp = fopen(filename, "r");
    if (fp == NULL)
    {
        perror("Error opening file");
        system("pause");
        exit(EXIT_FAILURE);
    }

    // Determine the actual number of rows and columns
    while (fscanf(fp, "%lf,", &temp) != EOF)
    {
        temp_col_count++;
        char c = fgetc(fp);
        if (c == '\n' || c == EOF)
        {
            if (actual_rows == 0)
            {
                actual_cols = temp_col_count;
            }
            else if (temp_col_count != actual_cols)
            {
                fprintf(stderr, "Error: Inconsistent number of columns in row %d\n", actual_rows + 1);
                fclose(fp);
                system("pause");
                exit(EXIT_FAILURE);
            }
            actual_rows++;
            temp_col_count = 0;
        }
    }

    if (actual_rows != dimension || actual_cols != dimension)
    {
        fprintf(stderr, "Error: Matrix dimensions in file (%d x %d) do not match expected dimensions (%d x %d)\n", actual_rows, actual_cols, dimension, dimension);
        fclose(fp);
        system("pause");
        exit(EXIT_FAILURE);
    }

    // Reset the file pointer to the beginning of the file
    rewind(fp);

    // Read the matrix data
    for (row = 0; row < dimension; row++)
    {
        for (col = 0; col < dimension; col++)
        {
            if (fscanf(fp, "%lf,", &L[row * dimension + col]) == EOF)
            {
                fprintf(stderr, "Error: Unexpected end of file while reading matrix data\n");
                fclose(fp);
                system("pause");
                exit(EXIT_FAILURE);
            }
        }
    }

    // Close the file
    fclose(fp);
    return 0;
}
