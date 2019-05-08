#include <math.h>

#include "activation_function.h"

double dx = 0.00001;

double ActivationFunction::sigmoid(const double z)
{
    return 1.0/(1.0+exp(-z));
}

double ActivationFunction::sigmoid_derivative(double z)
{
    return (this->sigmoid(z)-this->sigmoid(z-dx))/dx;
}