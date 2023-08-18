#ifndef HC_HEADER
#define HC_HEADER

#include "../../math/include/math.hpp"
#include <random>

namespace HC
{
    struct HCParms
    {
        int Lx, Ly, Lz, Lt;
        double Sx, Sy, Sz, St;
        double dx, dy, dz, dt;
        double amp;
    };

    namespace CPU
    {
        std::default_random_engine generator;
        std::normal_distribution<double> distribution;

        void Diff(double *dT, double *dQ, double *T, double *Q, HCParms &parms);

        void AllocWorkspaceEuler(double *&workspace, HCParms &parms);
        void FreeWorkspaceEuler(double *workspace);
        void Euler(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRK4(double *&workspace, HCParms &parms);
        void FreeWorkspaceRK4(double *workspace);
        void RK4(double *T, double *Q, double *workspace, HCParms &parms);

        void AllocWorkspaceRKF45(double *&workspace, HCParms &parms);
        void FreeWorkspaceRKF45(double *workspace);
        void RKF45(double *T, double *Q, double *workspace, HCParms &parms);
    }
    namespace GPU
    {

    }
}

#endif