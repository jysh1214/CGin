#ifndef UNIT_TEST_H
#define UNIT_TEST_H

#include <iostream>

#include <map>

#include "../src/activation_function.h"
#include "../src/fully_connected_layer.h"
#include "../src/matrix.h"
#include "../src/neural_network.h"
#include "../src/random.h"

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

class Dice_Recognize_TEST: public::testing::Test
{
protected:
    void SetUp() override
    {
        one = new double[9]{0,0,0,0,1,0,0,0,0};
        two = new double[9]{0,0,1,0,0,0,1,0,0};
        three = new double[9]{1,0,0,0,1,0,0,0,1};
        four = new double[9]{1,0,1,0,0,0,1,0,1};
        five = new double[9]{1,0,1,0,1,0,1,0,1};
        six = new double[9]{1,0,1,1,0,1,1,0,1};

        input_data.push_back(one);
        input_data.push_back(two);
        input_data.push_back(three);
        input_data.push_back(four);
        input_data.push_back(five);
        input_data.push_back(six);

        annotation.push_back(1);
        annotation.push_back(2);
        annotation.push_back(3);
        annotation.push_back(4);
        annotation.push_back(5);
        annotation.push_back(6);
    }

    void TearDown() override
    {
        delete one;
        delete two;
        delete three;
        delete four;
        delete five;
        delete six;
    }

    double * one;
    double * two;
    double * three;
    double * four;
    double * five;
    double * six;
    vector<double*> input_data;
    vector<int> annotation;
};

// class ActivationFunction TEST
TEST (ActivationFunction_sigmoid_TEST, )
{
    ActivationFunction af;
    ASSERT_EQ(1, af.sigmoid(200));
    ASSERT_TRUE(af.sigmoid(-3) < 1);
}

TEST (ActivationFunction_sigmoid_derivative_TEST, )
{
    ActivationFunction af;
    ASSERT_NEAR(0, af.sigmoid_derivative(200), 0.001);
    ASSERT_NEAR(-0.13439, af.sigmoid_derivative(1.235)* (-0.77), 0.001);
}

// class FullyConnectedLayer TEST
TEST (FullyConnectedLayer_Test, )
{
    FullyConnectedLayer f = FullyConnectedLayer(3, 9, 5, 6);

    struct matrix<double> m1 = f.getWeight(0);
    for (int i = 0; i < 3; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(1, m1.data[i][j]);
        }
    }

    struct matrix<double> m2 = f.getWeight(1);
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            ASSERT_EQ(1, m2.data[i][j]);
        }
    }
}

// class Random TEST
TEST (Random_TEST, )
{
    Random r;
    double random_num = r.GaussianDistribution(0.0, 1.0);
    ASSERT_TRUE(random_num < 1);
    ASSERT_TRUE(random_num + 1 > 0);
}

// MAIN TEST
TEST_F (Dice_Recognize_TEST, )
{
    FullyConnectedLayer f = FullyConnectedLayer(3, 9, 10, 6);
    f.GradientDescent(input_data, annotation, 10, 0.003);
    ASSERT_TRUE(1);
}

#endif
