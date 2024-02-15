#ifndef HC_HEADER
#define HC_HEADER

#include "../../math/include/math.hpp"
#include <random>
#include <curand.h>

#define ILSA 1
#define SIGMA 5.67e-8

namespace HC
{
    struct HCParms
    {
        int Lx, Ly, Lz, Lt;
        double Sx, Sy, Sz, St;
        double dx, dy, dz, dt;
        double amp, T_amb, T_ref, eps;
    };

    __host__ __device__ inline double C(double T_i);
    __host__ __device__ inline double K(double T_i);
    
    namespace RM
    {
        namespace CPU
        {
            void EvolutionJacobianMatrix(double *mTT, double *mTQ, double *mQT, double *mQQ, HCParms &parms);
            void EvaluationMatrix(double *mTT, double *mQT, HCParms &parms);
        };

        namespace GPU
        {
            void EvolutionJacobianMatrix(double *mTT, double *mTQ, double *mQT, double *mQQ, HCParms &parms);
            void EvaluationMatrix(double *mTT, double *mQT, HCParms &parms);
        };
    };

    namespace CPU
    {
        void Diff(double *dT, double *dQ, double *T, double *Q, HCParms &parms);

        void AllocWorkspaceEuler(double *&workspace, HCParms &parms);
        void FreeWorkspaceEuler(double *workspace);
        void Euler(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRK4(double *&workspace, HCParms &parms);
        void FreeWorkspaceRK4(double *workspace);
        void RK4(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRKF45(double *&workspace, HCParms &parms);
        void FreeWorkspaceRKF45(double *workspace);
        void ButcherTableau(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref0, double *ref1, int L0, int L1, HCParms &parms);
        void RKF45(double *T, double *Q, double *workspace, HCParms &parms);
    }
    namespace GPU
    {
        void Diff(double *dT, double *dQ, double *T, double *Q, HCParms &parms);

        void AllocWorkspaceEuler(double *&workspace, HCParms &parms);
        void FreeWorkspaceEuler(double *workspace);
        void Euler(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRK4(double *&workspace, HCParms &parms);
        void FreeWorkspaceRK4(double *workspace);
        void RK4(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRKF45(double *&workspace, HCParms &parms);
        void FreeWorkspaceRKF45(double *workspace);
        void ButcherTableau(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref0, double *ref1, int L0, int L1, HCParms &parms);
        void RKF45(double *T, double *Q, double *workspace, HCParms &parms);
    }
}

#endif