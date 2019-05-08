#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

class ActivationFunction
{
public:
    ActivationFunction() {}
    virtual ~ActivationFunction() {}

    double sigmoid(double z);
    double sigmoid_derivative(double z);

private:
};

#endif