#include <stdlib.h>
#include <time.h>
#include <math.h>
#define PI 3.14159

#include "random.h"

int x = 0;

double Random::GaussianDistribution(const double mean, const double variance)
{
    double A, B, C, r;
    double uni[2];
    srand((unsigned)time(NULL));

    UNIFORM(&uni[0]);
    A = sqrt((-2)*log(uni[0]));
    B = 2 * PI*uni[1];
    C = A*cos(B);
    r = mean + C*variance;

    return r;
}

void Random::UNIFORM(double * p)
{
    int i, a;
    double f;
    for (i = 0; i<2; i++, x = x+689)
    {
        a = rand() + x;
        a = a%1000;
        f = (double)a;
        f = f/1000.0;
        *p = f;
        p++;
    }
}