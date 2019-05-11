#ifndef ACTIVATION_FUNCTION
#define ACTIVATION_FUNCTION

class ActivationFunction
{
public:
    ActivationFunction() {}
    virtual ~ActivationFunction() 
    {
        // af_ptr = &this->sigmoid;
    }

    double sigmoid(const double z);
    double sigmoid_derivative(const double z);
    double derivative(double (ActivationFunction::*af_ptr)(double), double z);

    // ActivationFunction * af_ptr;

private:
};

#endif