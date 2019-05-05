#ifndef RANDOM_H
#define RANDOM_H

using namespace std;

class Random
{
public:
    Random() {}
    virtual ~Random() {}

    double GaussianDistribution(const double mean, const double variance);

private:
    void UNIFORM(double * p);
};

#endif