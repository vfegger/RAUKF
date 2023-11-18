#ifndef RANDOMGPU_HEADER
#define RANDOMGPU_HEADER

#include <curand.h>

namespace RandomGPU
{
    // Sample Random Vector with Normal Distribution
    void SampleNormal(double* v_o, int length, double mean, double sigma);
}


#endif