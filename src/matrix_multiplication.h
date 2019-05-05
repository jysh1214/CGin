#ifndef MATRIX_MULTIPLICATION_H
#define MATRIX_MULTIPLICATION_H

#include <string>
#include "matrix.h"

using namespace std;

/*
* a * b = c 
*
* @return c
*/
template <class T>
void matrixMultiplication(struct matrix<T> &a, struct matrix<T> &b, struct matrix<T> &c)
{
    if (a.col != b.row || a.row != c.row || b.col != c.col)
        throw string("matrix multiplication error\n");
}

#endif