#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

class ActivationFunction
{
public:
    ActivationFunction() {}
    virtual ~ActivationFunction() {}

    double sigmoid(const double z);
    double sigmoid_derivative(const double z);

private:
};

#endif