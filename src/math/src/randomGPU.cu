#include "../include/randomGPU.hpp"

void RandomGPU::SampleNormal(double *v_o, int length, double mean, double sigma)
{
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetStream(generator,cudaStreamDefault);
    curandGenerateNormalDouble(generator,v_o,length,mean,sigma);
    curandDestroyGenerator(generator);
}