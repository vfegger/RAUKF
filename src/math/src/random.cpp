#include "../include/random.hpp"

void Random::SampleNormal(Pointer<double> v_o, int length, double mean, double sigma, Type type)
{
    if (type == Type::CPU)
    {
        RandomCPU::SampleNormal(v_o.host(), length, mean, sigma);
    }
    else if (type == Type::GPU)
    {
        RandomGPU::SampleNormal(v_o.dev(), length, mean, sigma);
    }
}