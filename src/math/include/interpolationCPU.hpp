#ifndef INTERPOLATIONCPU_HEADER
#define INTERPOLATIONCPU_HEADER

namespace InterpolationCPU
{
    inline double Lerp(double x, double x0, double x1, double v0, double v1);

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