#ifndef MATRIX_H
#define MATRIX_H

#include <vector>

using namespace std;

template <class T>
struct matrix
{
    // struct proxy
    // {
    //     vector<T>* vec;
    //     proxy(vector<T>* vec_): vec(vec_) {}

    //     T& operator[](unsigned int index)
    //     {
    //         return (*vec)[index];
    //     }
    // };
    
    matrix() = default;
    matrix(unsigned int r, unsigned int c): row(r), col(c), data(r, vector<T>(c)) {}

    // proxy operator[](int index)
    // {
    //     return proxy(&data[index]);
    // }

    unsigned int row, col;
    vector<vector<T>> data;
};

#endif