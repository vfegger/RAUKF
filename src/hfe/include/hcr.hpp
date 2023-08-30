#ifndef HCR_HEADER
#define HCR_HEADER

#include "../../math/include/math.hpp"
#include <random>

#define SIGMA 5.67e-8

namespace HCR
{
    struct HCRParms
    {
        int Lx, Ly, Lz, Lt;
        double Sx, Sy, Sz, St;
        double dx, dy, dz, dt;
        double amp;
    };

    namespace CPU
    {
        static std::default_random_engine generator;
        static std::normal_distribution<double> distribution;

        void Diff(double *dT, double *dQ, double *T, double *Q, HCRParms &parms);

        void AllocWorkspaceEuler(double *&workspace, HCRParms &parms);
        void FreeWorkspaceEuler(double *workspace);
        void Euler(double *T, double *Q, double *workspace, HCRParms &parms);

        void AllocWorkspaceRK4(double *&workspace, HCRParms &parms);
        void FreeWorkspaceRK4(double *workspace);
        void RK4(double *T, double *Q, double *workspace, HCRParms &parms);

        void AllocWorkspaceRKF45(double *&workspace, HCRParms &parms);
        void FreeWorkspaceRKF45(double *workspace);
        void ButcherTableau(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref0, double *ref1, int L0, int L1, HCRParms &parms);
        void RKF45(double *T, double *Q, double *workspace, HCRParms &parms);
    }
    namespace GPU
    {

    }
}

#endif