#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <string>
#include "matrix.h"

using namespace std;

/*
* A * B = C 
*
* @return C
*/
template <class T>
void matrixMultiplication(struct matrix<T> &A, struct matrix<T> &B, struct matrix<T> &C)
{
    if (A.col != B.row || A.row != C.row || B.col != C.col)
        throw string("matrix multiplication error\n");

    unsigned int N = A.row;
    unsigned int M = B.row;
    unsigned int R = C.col;

    for (unsigned int i=0; i<N; i++)
    {
        for (unsigned int j=0; j<R; j++)
        {
            C.data[i][j] = 0;
            for (unsigned int k=0; k<M; k++)
                C.data[i][j] += A.data[i][k] * B.data[k][j];
        }
    }
}

#endif