#ifndef HC_HEADER
#define HC_HEADER

#include "../../math/include/math.hpp"
#include <random>
#include <curand.h>

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
        static std::default_random_engine generator;
        static std::normal_distribution<double> distribution;
        static double *noise;

        void AddNoise(double *Q, HCParms &parms);
        void SetNoise(HCParms &parms);
        void UnsetNoise();

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
        static curandGenerator_t generator;
        static double *noise;
        static long long unsigned offset = 0llu;

        void AddNoise(double *Q, HCParms &parms);
        void SetNoise(HCParms &parms);
        void UnsetNoise();

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