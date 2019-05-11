#include <math.h>

#include "activation_function.h"

double dx = 0.000001;

double ActivationFunction::sigmoid(const double z)
{
    return 1.0/(1.0+exp(-z));
}

double ActivationFunction::sigmoid_derivative(const double z)
{
    return (this->sigmoid(z)-this->sigmoid(z-dx))/dx;
}

double ActivationFunction::derivative(double (ActivationFunction::*af_ptr)(double), double z)
{
    return ((this->*af_ptr)(z)-(this->*af_ptr)(z-dx))/dx;
}