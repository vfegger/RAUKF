#ifndef HC2D_HEADER
#define HC2D_HEADER

#include "../../math/include/math.hpp"
#include <random>
#include <curand.h>

namespace HC2D
{
    struct HCParms
    {
        int Lx, Ly, Lt;
        double Sx, Sy, Sz, St;
        double dx, dy, dt;
        double amp, h;
    };

    __host__ __device__ inline double C(double x, double y);
    __host__ __device__ inline double K(double x, double y);

    namespace CPU
    {
        void EvolutionJacobianMatrix(double *mTT, double *mTQ, double *mQT, double *mQQ, HCParms &parms);
        void EvolutionControlMatrix(double *mUT, double *mUQ, HCParms &parms);
        void EvaluationMatrix(double *mTT, double *mQT, HCParms &parms);
    };

    namespace GPU
    {
        void EvolutionJacobianMatrix(double *mTT, double *mTQ, double *mQT, double *mQQ, HCParms &parms);
        void EvolutionControlMatrix(double *mUT, double *mUQ, HCParms &parms);
        void EvaluationMatrix(double *mTT, double *mQT, HCParms &parms);
    };
}

#endif