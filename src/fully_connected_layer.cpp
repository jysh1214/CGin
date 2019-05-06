#include <iostream>

#include <cstdarg>
#include <math.h>
#include <map>

#include "fully_connected_layer.h"

#include "matrix.h"
#include "matrix_multiplication.h"
#include "random.h"

using namespace std;

/*
* @param number_of_layers: the number of layers
* @param: the number of neurons of first layer
* @param: the number of neurons of second layer ...
*
* @return: none
*/
FullyConnectedLayer::FullyConnectedLayer(const unsigned int number_of_layers, ...): number_of_layers(0)
{
    if (number_of_layers < 2) throw string("number of layers must be greater than 1\n");
    this->number_of_layers += number_of_layers;
    this->number_of_neurons_for_each_layer = (int*) new int[number_of_layers];

    va_list number_of_neurons_for_each_layer;
    va_start(number_of_neurons_for_each_layer, number_of_layers);
        for (unsigned int i = 0; i < number_of_layers; i++)
        {
            this->number_of_neurons_for_each_layer[i] = 
                va_arg(number_of_neurons_for_each_layer, int);
        }
            
    va_end(number_of_neurons_for_each_layer);

    // both the biases and the weight are randomly initialized
    this->biases = new matrix<double>[number_of_layers - 1];
    this->weights = new matrix<double>[number_of_layers - 1];
    Random random;
    for (unsigned int i = 0; i < number_of_layers - 1; i++)
    {
        // number of neurons of first layer
        unsigned int r = this->number_of_neurons_for_each_layer[i];
        // number of neurons of second layer
        unsigned int l = this->number_of_neurons_for_each_layer[i+1];
        
        matrix<double> biases_matrix(1, l);
        for (unsigned int i = 0; i < l; i++)
            biases_matrix.data[0][i] = random.GaussianDistribution(0.0, 1.0);
        this->biases[i] = biases_matrix;

        matrix<double> weights_matrix(r, l);
        for (unsigned int j = 0; j < r; j++)
        {
            for (unsigned int k = 0; k < l; k++)
                weights_matrix.data[j][k] = random.GaussianDistribution(0.0, 1.0);
        }
        this->weights[i] = weights_matrix;
    }
}

void FullyConnectedLayer::setActivationFunction(const ActivationFunction activation_function)
{

}

void FullyConnectedLayer::addHidenLayer(const FullyConnectedLayer fully_connected_layer)
{
    this->hidenLayer.push_back(fully_connected_layer);
    this->number_of_layers ++;
}

/*
* @param which_weight: choose which weigth you want
* @return which weight you want 
*/
struct matrix<double> FullyConnectedLayer::getWeight(const int which_weight)
{
    return weights[which_weight];
}

void FullyConnectedLayer::GradientDescent(const vector<double*> &input_data, 
    const vector<int> &annotation, const unsigned int epoch, double learning_rate, unsigned int mini_batch_size)
{
    unsigned int total_batch_size = 0;
    unsigned int epoch_count = 0;

    double loss_sum = 0;

    unsigned int first_data_length = this->number_of_neurons_for_each_layer[0];
    unsigned int last_data_length = this->number_of_neurons_for_each_layer[this->number_of_layers];

    for (; epoch_count < epoch; epoch_count++)
    {

    vector<double*>::const_iterator it = input_data.begin();
    vector<int>::const_iterator an_it = annotation.begin();
    for (; total_batch_size < input_data.size(); total_batch_size += mini_batch_size) // mini_batch_size = 1
    {
        // convert double * to struct matrix *
        this->forward_matrix = new matrix<double>(1, first_data_length);
        for (unsigned int i = 0; i < first_data_length; i++) 
        {
            this->forward_matrix->data[0][i] = (*it)[i];
        }
        this->forward(0);
        it ++;

        // count loss
        for (unsigned int i = 0; i < last_data_length; i++)
        {
            if (i == *an_it) // output should be 1
                loss_sum += (1-this->forward_matrix->data[0][i])*(1-this->forward_matrix->data[0][i]);
            else // output should be 0
                loss_sum += (this->forward_matrix->data[0][i])*(this->forward_matrix->data[0][i]);
        }
        an_it ++;

    } // batch data iterator

    //帶入loss function 做梯地下降 調整權重和偏至
    loss_sum /= last_data_length;

    // vis
    cout<< "第" <<epoch_count+1<< "次迭代： " << "loss: "<< loss_sum<<endl;


    //全部迭代完成後 為一epoch

    } // epoch iterator
}

/*
* matrix multiplaication:
* A * B = C
* A: 1*input_dimension, B: input_dimension*output_dimension, C: 1*output_dimension
*
* @param double * patch_data: input data
* @param int data_length: input data length
* @param int which_layer: from 0 to (number of layers -1)
*/
void FullyConnectedLayer::forward(const int which_layer)
{
    if (which_layer >= this->number_of_layers) throw string("out of range");
    if (which_layer == this->number_of_layers - 1) return;

    unsigned int input_dimension;
    unsigned int output_dimension;

    input_dimension = number_of_neurons_for_each_layer[which_layer];
    output_dimension = number_of_neurons_for_each_layer[which_layer + 1];

    struct matrix<double> A(1, input_dimension);
    struct matrix<double> B(input_dimension, output_dimension);
    struct matrix<double> C(1, output_dimension);

    // read forward matrix
    for (unsigned int i = 0; i < input_dimension; i++)
        A.data[0][i] = this->forward_matrix->data[0][i];

    // read weigths
    B = this->getWeight(which_layer);

    // matrix multiplication
    matrixMultiplication(A, B, C);

    // assign answer to forward matrix
    free(this->forward_matrix);
    this->forward_matrix = new matrix<double>(1, output_dimension);
    ActivationFunction af;
    for (unsigned int i = 0; i < output_dimension; i++)
    {
        this->forward_matrix->data[0][i] = 
            C.data[0][i] + this->biases[which_layer].data[0][i];

        this->forward_matrix->data[0][i] = 
            af.sigmoid(this->forward_matrix->data[0][i]);
    }
        
    this->forward(which_layer+1);
}

/*
* @return the loss of current weights
*/
double FullyConnectedLayer::loss_function()
{
    return 1.11;
}

// TEST
void FullyConnectedLayer::showWeights(int layer)
{
    // number of neurons of first layer
    unsigned int r = this->number_of_neurons_for_each_layer[layer];
    // number of neurons of second layer
    unsigned int l = this->number_of_neurons_for_each_layer[layer+1];
    
    // cout<<r<<" "<<l<<endl; // terminal test
    for (unsigned int i = 0; i < r; i++)
    {
        for (unsigned int j = 0; j < l; j++)
        {
            // cout<<this->weights[layer].data[i][j]<<", ";
        }
        // cout<<endl;
    }
    // cout<<endl;
}