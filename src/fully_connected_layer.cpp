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
    this->number_of_neurons_for_each_layer = new int[number_of_layers];

    va_list number_of_neurons_for_each_layer;
    va_start(number_of_neurons_for_each_layer, number_of_layers);
        for (unsigned int i = 0; i < number_of_layers; i++)
        {
            this->number_of_neurons_for_each_layer[i] = va_arg(number_of_neurons_for_each_layer, int);
            if (this->number_of_neurons_for_each_layer[i] <= 0)
                throw string("number of layers is illegal\n");
        }
    va_end(number_of_neurons_for_each_layer);

    // both the biases and the weight are randomly initialized
    this->biases = new matrix<double>[number_of_layers - 1];
    // this->weights = new matrix<double>(1, number_of_layers - 1);
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

        matrix<double> * weights_matrix = new matrix<double>(r, l);
        for (unsigned int j = 0; j < r; j++)
        {
            for (unsigned int k = 0; k < l; k++)
                weights_matrix->data[j][k] = random.GaussianDistribution(0.0, 1.0);
        }
        this->weights.push_back(weights_matrix);
    }
}

void FullyConnectedLayer::GradientDescent(const vector<double*> &input_data, 
    const vector<int> &annotation, const unsigned int epoch, double learning_rate, unsigned int mini_batch_size)
{
    unsigned int epoch_count = 0;
    unsigned int total_batch_size = 0;

    unsigned int first_data_length = this->number_of_neurons_for_each_layer[0];
    unsigned int last_data_length = this->number_of_neurons_for_each_layer[this->number_of_layers-1];

    double loss_sum;

    for (; epoch_count < epoch; epoch_count++)
    {

    loss_sum = 0;

    vector<double*>::const_iterator in_it = input_data.begin();
    vector<int>::const_iterator an_it = annotation.begin();
    for (; total_batch_size < input_data.size(); total_batch_size += mini_batch_size) // mini_batch_size = 1
    {
        // convert double * to struct matrix *
        this->forward_matrix = new matrix<double>(1, first_data_length);

        for (unsigned int i = 0; i < first_data_length; i++)
        {
            this->forward_matrix->data[0][i] = (*in_it)[i];
        }
        this->cells_value.push_back(this->forward_matrix);
        this->forward(0);

        // adjust weight
        this->adjust_weights(*an_it);
        this->cells_value.clear();

        in_it ++;

        // count loss for this batch size
        for (int i = 0; i < signed(last_data_length); i++)
        {
            if (i == *an_it) // output should be 1
            {
                loss_sum += (1-this->forward_matrix->data[0][i])*(1-this->forward_matrix->data[0][i]);
            }
            else // output should be 0
            {
                loss_sum += (this->forward_matrix->data[0][i])*(this->forward_matrix->data[0][i]);
            }
        }
        an_it ++;
        free(this->forward_matrix);
    } // batch data iterator
    loss_sum /= last_data_length;
    // visualization
    cout<< "第" << epoch_count+1 << "次迭代： " << "loss: " << loss_sum << endl;
    
    total_batch_size = 0;
    } // epoch iterator
}

/*
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

    struct matrix<double> * A = new matrix<double>(1, input_dimension);
    struct matrix<double> * B = new matrix<double>(input_dimension, output_dimension);
    struct matrix<double> * C = new matrix<double>(1, output_dimension);

    // read forward matrix
    for (unsigned int i = 0; i < input_dimension; i++)
        A->data[0][i] = this->forward_matrix->data[0][i];

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
        this->forward_matrix->data[0][i] = C->data[0][i];
            //C.data[0][i] + this->biases[which_layer].data[0][i];
    }

    // save origin cells value
    this->cells_value.push_back(this->forward_matrix);
    
    // activation function
    for (unsigned int i = 0; i < output_dimension; i++)
    {
        this->forward_matrix->data[0][i] = 
            af.sigmoid(this->forward_matrix->data[0][i]);
    }

    this->forward(which_layer+1);
}

void FullyConnectedLayer::adjust_weights(const double target)
{
    ActivationFunction af;
    int output_length = number_of_neurons_for_each_layer[this->number_of_layers-1];
    double  delta_output_sum;
    double cell_value;
    double result;
    double delta_weight; int j = this->number_of_layers-1;

    matrix<double> * hidden_layer_results = nullptr;
    for (int i = 0; i < output_length; i++)
    {
        // for (int j = this->number_of_layers-1; j > 0; j--)
        // {
            cell_value = (this->cells_value[j])->data[0][i];
            result = af.sigmoid(cell_value);
            delta_output_sum = af.sigmoid_derivative(cell_value) * (target - result);
            // cout<< delta_output_sum <<endl;

            hidden_layer_results = this->cells_value[j-1];

            // delta_weights = delta_output_sum / hidden_layer_results;
            // number_of_neurons = number of neurons of last layer
            unsigned int number_of_neurons = number_of_neurons_for_each_layer[j-1];
            for (unsigned int k = 0; k < number_of_neurons; k++)
            {
                (hidden_layer_results->data[0][k] == 0)?
                delta_weight = 0 :
                delta_weight = delta_output_sum / hidden_layer_results->data[0][k];
                // cout<<hidden_layer_results->data[0][k]<<endl;
                // cout<<delta_weight * 10<<endl;

                // new weights = old weights + delta_weights
                this->weights[j-1]->data[k][j] += delta_weight;
            }
        // } // layer iterator
    } // output result iterator
}

/*
* @param which_weight: choose which weigth you want
* @return which weight you want
*/
struct matrix<double> * FullyConnectedLayer::getWeight(const int which_weight)
{
    return weights[which_weight];
}
