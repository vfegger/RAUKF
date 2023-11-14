#include "..\include\interpolation.hpp"

void Interpolation::Rescale(Pointer<double> p_i, int Lx_i,
             Pointer<double> p_o, int Lx_o,
             double Sx, Type type)
{
    if (type == Type::CPU)
    {
        InterpolationCPU::Rescale(p_i.host(), Lx_i, p_o.host(), Lx_o, Sx);
    }
    else if (type == Type::GPU)
    {
        InterpolationGPU::Rescale(p_i.dev(), Lx_i, p_o.dev(), Lx_o, Sx);
    }
}
void Interpolation::Rescale(Pointer<double> p_i, int Lx_i, int Ly_i,
             Pointer<double> p_o, int Lx_o, int Ly_o,
             double Sx, double Sy, Type type)
{
    if (type == Type::CPU)
    {
        InterpolationCPU::Rescale(p_i.host(), Lx_i, Ly_i, p_o.host(), Lx_o, Ly_o, Sx, Sy);
    }
    else if (type == Type::GPU)
    {
        InterpolationGPU::Rescale(p_i.dev(), Lx_i, Ly_i, p_o.dev(), Lx_o, Ly_o, Sx, Sy);
    }
}
void Interpolation::Rescale(Pointer<double> p_i, int Lx_i, int Ly_i, int Lz_i,
             Pointer<double> p_o, int Lx_o, int Ly_o, int Lz_o,
             double Sx, double Sy, double Sz, Type type)
{
    if (type == Type::CPU)
    {
        InterpolationCPU::Rescale(p_i.host(), Lx_i, Ly_i, Lz_i, p_o.host(), Lx_o, Ly_o, Lz_o, Sx, Sy, Sz);
    }
    else if (type == Type::GPU)
    {
        InterpolationGPU::Rescale(p_i.dev(), Lx_i, Ly_i, Lz_i, p_o.dev(), Lx_o, Ly_o, Lz_o, Sx, Sy, Sz);
    }
}
