#include <math.h>

#include "activation_function.h"

double ActivationFunction::sigmoid(const double z)
{
    return 1.0/(1.0+exp(-z));
}