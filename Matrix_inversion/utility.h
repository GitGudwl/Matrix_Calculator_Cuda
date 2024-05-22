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
			plik << A[j * n + i] << "\t";
		}
		plik << endl;
	}
	plik.close();
}

void matrix_read(double* L, int dimension, const char* filename)
{
    FILE* fp;
    int row, col;

    fp = fopen(filename, "r"); // open the file with the given filename
    if (fp == NULL)            // open failed
        return;

    for (row = 0; row < dimension; row++)
    {
        for (col = 0; col < dimension; col++)
            if (fscanf(fp, "%lf,", &L[row * dimension + col]) == EOF)
                break; // read data

        if (feof(fp))
            break; // if the file is over
    }

    fclose(fp); // close file
}
