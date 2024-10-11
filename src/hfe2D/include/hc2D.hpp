#ifndef HC2D_HEADER
#define HC2D_HEADER

#include "../../math/include/math.hpp"
#include <random>
#include <curand.h>

#define IMPLICIT_SCHEME 1

namespace HC2D
{
    struct HCParms
    {
        int Lx, Ly, Lt;
        double Sx, Sy, Sz, St;
        double dx, dy, dt;
        double amp, h;

        bool operator==(const HCParms &rhs)
        {
            return (this->Lx == rhs.Lx) && (this->Ly == rhs.Ly) && (this->Lt == rhs.Lt) &&
                   (this->Sx == rhs.Sx) && (this->Sy == rhs.Sy) && (this->Sz == rhs.Sz) && (this->St == rhs.St) &&
                   (this->dx == rhs.dx) && (this->dy == rhs.dy) && (this->dt == rhs.dt) &&
                   (this->amp == rhs.amp) && (this->h == rhs.h);
        }
    };

    __host__ __device__ inline double C(double x, double y);
    __host__ __device__ inline double K(double x, double y);

    void validate(HCParms &parms);
    static HCParms refparms;
    static Pointer<double> AI;  // Matrix A Implicit
    static Pointer<double> BE;  // Matrix B Explicit
    static Pointer<double> CE;  // Matrix C Explicit
    static Pointer<double> ATA; // Matrix A^T*A to solve a LP
    static Pointer<double> JX;  // Matrix J over X
    static Pointer<double> JU;  // Matrix J over U
    static bool isValid = false;

    namespace CPU
    {
        void ImplicitScheme(HCParms &parms, int strideTQ);
        void ExplicitScheme(HCParms &parms, int strideTQ);
        void EvolutionMatrix(HCParms &parms, double *pmXX_o, double *pmUX_o, int strideTQ);
        void EvaluationMatrix(HCParms &parms, double *mTT, double *mQT);
    };

    namespace GPU
    {
        void ImplicitScheme(HCParms &parms, int strideTQ);
        void ExplicitScheme(HCParms &parms, int strideTQ);
        void EvolutionMatrix(HCParms &parms, double *pmXX_o, double *pmUX_o, int strideTQ);
        void EvaluationMatrix(HCParms &parms, double *mTT, double *mQT);
    };
}

#endif