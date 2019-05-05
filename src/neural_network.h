#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <string>
#include "activation_function.h"

using namespace std;

class NeuralNetwork
{
public:
    virtual void setActivationFunction(string activation_function)
    {
        throw string("unable to use\n");
    }
};

#endif