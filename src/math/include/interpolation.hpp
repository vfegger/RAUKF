#ifndef INTERPOLATION_HEADER
#define INTERPOLATION_HEADER

#include "../../structure/include/pointer.hpp"
#include "interpolationCPU.hpp"
#include "interpolationGPU.hpp"

namespace Interpolation
{
    void Rescale(Pointer<double> p_i, int Lx_i,
                 Pointer<double> p_o, int Lx_o,
                 double Sx, Type type);
    void Rescale(Pointer<double> p_i, int Lx_i, int Ly_i,
                 Pointer<double> p_o, int Lx_o, int Ly_o,
                 double Sx, double Sy, Type type);
    void Rescale(Pointer<double> p_i, int Lx_i, int Ly_i, int Lz_i,
                 Pointer<double> p_o, int Lx_o, int Ly_o, int Lz_o,
                 double Sx, double Sy, double Sz, Type type);

}

#endif