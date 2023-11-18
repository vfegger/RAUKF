#ifndef RANDOM_HEADER
#define RANDOM_HEADER

#include "../../structure/include/pointer.hpp"
#include "randomCPU.hpp"
#include "randomGPU.hpp"

namespace Random
{
    // Sample Random Vector with Normal Distribution
    void SampleNormal(Pointer<double> v_o, int length, double mean, double sigma, Type type);
}

#endif