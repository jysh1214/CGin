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
    if (number_of_layers < 2) throw string("number of layers must be greater than 2\n");
    this->number_of_layers += number_of_layers;
    this->number_of_neurons_for_each_layer = (int*) new int[number_of_layers];

    va_list number_of_neurons_for_each_layer;
    va_start(number_of_neurons_for_each_layer, number_of_layers);
        for (unsigned int i = 0; i < number_of_layers; i++)
            this->number_of_neurons_for_each_layer[i] = va_arg(number_of_neurons_for_each_layer, int);
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

// void FullyConnectedLayer::GradientDescent(const map<const double*, int, ptr_less<const double>> input_data, 
//                                             int epoch, int mini_batch_size)
// {
//     // cout<<input_data.size()<<endl;
//     for (auto &item:input_data) cout<<item.first<<": "<<item.second<<endl;
// }

void FullyConnectedLayer::GradientDescent(const vector<double*> &input_data, const vector<int> &annotation, 
                                            const int epoch, unsigned int mini_batch_size)
{
    // int first_layer = number_of_neurons_for_each_layer[0]; // input data length
    unsigned int total_batch_size = 0;
    //batch-size大小的data 進入 直到底
    vector<double*>::const_iterator it = input_data.begin();
    for (; total_batch_size < input_data.size(); total_batch_size += mini_batch_size) // mini_batch_size = 1
    {
        // unsigned int data_length = this->number_of_neurons_for_each_layer[0];
        // matrix<double> cell(1, data_length);
        // for (unsigned int i = 0; i < data_length; i++) 
        // {
        //     cell.data[0][i] = (*it)[i];
        // }

        // this->forward(cell, 0);

        this->forward(*it, 0);
        it ++;
    }
    

    //帶入loss function 做梯地下降 調整權重和偏至

    //全部迭代完成後 為一epoch
}

/*
* @param double * patch_data: input data
* @param int data_length: input data length
* @param int which_layer: from 0 to (number of layers -1)
*/
void FullyConnectedLayer::forward(const double * cell, const int which_layer)
{
    if (which_layer >= this->number_of_layers) throw string("out of range");
    if (which_layer == this->number_of_layers - 1) return;

    // this layer
    matrix<double> weights_of_layer = this->getWeight(which_layer);
    // next layer
    matrix<double> answer(1, this->number_of_neurons_for_each_layer[which_layer+1]);

    // convert double * to matrix

    try
    {
        matrixMultiplication(cell, weights_of_layer, answer);
    }
    catch(const string e)
    {
        cerr << e << '\n';
    }

    for (int i = 0; i < answer.col; i++)
    {
        answer.data[0][i] += this->biases[which_layer+1].data[0][i];
    }

    // forward to next layer
    this->forward(answer, which_layer+1);
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