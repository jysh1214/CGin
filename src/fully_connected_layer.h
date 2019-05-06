#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include <vector>
#include <string>
#include "neural_network.h"
#include "activation_function.h"

#include "matrix.h"

using namespace std;

#ifndef ptr_less_
#define ptr_less_
template<class T> struct ptr_less 
{
    bool operator()(T* lhs, T* rhs) 
    {
        return *lhs < *rhs; 
    }
};
#endif

class FullyConnectedLayer: public NeuralNetwork
{
public:
    FullyConnectedLayer(const unsigned int number_of_layers, ...);
    virtual ~FullyConnectedLayer() {}
    void setActivationFunction(const ActivationFunction activation_function);
    void addHidenLayer(const FullyConnectedLayer fully_connected_layer);

    void GradientDescent(const vector<double*> &input_data, const vector<int> &annotation, 
                        int epoch, unsigned int mini_batch_size=1);  

    struct matrix<double> getWeight(const int which_weight);

    // TEST
    void showWeights(int i);
    
private:
    double loss_function();
    void forward(const int which_layer);

    vector <FullyConnectedLayer> hidenLayer;
    vector <NeuralNetwork> flow;

    int number_of_layers;
    int * number_of_neurons_for_each_layer;
    ActivationFunction activation_function;

    struct matrix<double> * biases;

    /* weight between two layers 
    * weight[0] : layer_1 to layer_2
    * weight[1] : layer_2 to layer_3 ...
    * 
    * weight:  0   1   2 ...
    * layer: 0| |1| |2| |3 ...
    */
    struct matrix<double> * weights;

    struct matrix<double> * forward_matrix;
};

#endif