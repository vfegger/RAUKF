#include "../include/randomCPU.hpp"

#include <random>

void RandomCPU::SampleNormal(double *v_o, int length, double mean, double sigma)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(mean, sigma);
    for (int i = 0; i < length; i++)
    {
        v_o[i] = distribution(generator);
    }
}