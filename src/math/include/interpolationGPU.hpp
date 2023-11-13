#ifndef INTERPOLATIONGPU_HEADER
#define INTERPOLATIONGPU_HEADER

#define CEIL(value, div) ((value + div - 1) / div)

namespace InterpolationGPU
{
    void Rescale(double *p_i, int Lx_i,
                 double *p_o, int Lx_o,
                 double Sx);
    void Rescale(double *p_i, int Lx_i, int Ly_i,
                 double *p_o, int Lx_o, int Ly_o,
                 double Sx, double Sy);
    void Rescale(double *p_i, int Lx_i, int Ly_i, int Lz_i,
                 double *p_o, int Lx_o, int Ly_o, int Lz_o,
                 double Sx, double Sy, double Sz);
}

#endif