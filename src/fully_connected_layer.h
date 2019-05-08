#ifndef FULLY_CONNECTED_LAYER
#define FULLY_CONNECTED_LAYER

#include <vector>
#include <string>
#include "neural_network.h"
#include "activation_function.h"

#include "matrix.h"

using namespace std;

class FullyConnectedLayer: public NeuralNetwork
{
public:
    FullyConnectedLayer() {}
    FullyConnectedLayer(const unsigned int number_of_layers, ...);
    virtual ~FullyConnectedLayer() {}

    void GradientDescent(const vector<double*> &input_data, const vector<int> &annotation, 
        const unsigned int epoch, double learning_rate, unsigned int mini_batch_size=1);  

private:
    void forward(const int which_layer);
    struct matrix<double> getWeight(const int which_weight);

    vector <FullyConnectedLayer> hidenLayer;
    vector <NeuralNetwork> flow;

    int number_of_layers;
    int * number_of_neurons_for_each_layer;
    ActivationFunction activation_function;

    /* 
    * weight between two layers 
    * weight[0] : layer_1 to layer_2
    * weight[1] : layer_2 to layer_3 ...
    * 
    * bias  :  0   1   2 ...
    * weight:  0   1   2 ...
    * layer: 0| |1| |2| |3 ...
    */
    struct matrix<double> * biases;
    struct matrix<double> * weights;

    // to store forward matrix
    struct matrix<double> * forward_matrix;
    vector <matrix<double>*> cells_value;
};

#endif