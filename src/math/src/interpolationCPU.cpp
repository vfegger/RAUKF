#include "../include/interpolationCPU.hpp"

#include <algorithm>

inline double InterpolationCPU::Lerp(double x, double x0, double x1, double v0, double v1)
{
    return (x1 - x) / (x1 - x0) * v0 + (x - x0) / (x1 - x0) * v1;
}

void InterpolationCPU::Rescale(double *p_i, int Lx_i,
                               double *p_o, int Lx_o,
                               double Sx)
{
    for (int i = 0; i < Lx_o; i++)
    {
        double x_o = (i - 0.5) * Sx / Lx_o;
        int i0_i = std::clamp((i * Lx_i) / Lx_o, 0, Lx_i - 2);
        int i1_i = std::clamp((i * Lx_i) / Lx_o + 1, 1, Lx_i - 1);
        double x0_i = (i0_i - 0.5) * Sx / Lx_i;
        double x1_i = (i1_i - 0.5) * Sx / Lx_i;
        p_o[i] = Lerp(x_o, x0_i, x1_i, p_i[i0_i], p_i[i1_i]);
    }
}

void InterpolationCPU::Rescale(double *p_i, int Lx_i, int Ly_i,
                               double *p_o, int Lx_o, int Ly_o,
                               double Sx, double Sy)
{
    for (int j = 0; j < Ly_o; j++)
    {
        double y_o = (j - 0.5) * Sy / Ly_o;
        int j0_i = std::clamp((j * Ly_i) / Ly_o, 0, Ly_i - 2);
        int j1_i = std::clamp((j * Ly_i) / Ly_o + 1, 1, Ly_i - 1);
        double y0_i = (j0_i - 0.5) * Sy / Ly_i;
        double y1_i = (j1_i - 0.5) * Sy / Ly_i;

        for (int i = 0; i < Lx_o; i++)
        {
            double x_o = (i - 0.5) * Sx / Lx_o;
            int i0_i = std::clamp((i * Lx_i) / Lx_o, 0, Lx_i - 2);
            int i1_i = std::clamp((i * Lx_i) / Lx_o + 1, 1, Lx_i - 1);
            double x0_i = (i0_i - 0.5) * Sx / Lx_i;
            double x1_i = (i1_i - 0.5) * Sx / Lx_i;

            p_o[j * Lx_o + i] = Lerp(y_o, y0_i, y1_i,
                                     Lerp(x_o, x0_i, x1_i, p_i[j0_i * Lx_i + i0_i], p_i[j0_i * Lx_i + i1_i]),
                                     Lerp(x_o, x0_i, x1_i, p_i[j1_i * Lx_i + i0_i], p_i[j1_i * Lx_i + i1_i]));
        }
    }
}

void InterpolationCPU::Rescale(double *p_i, int Lx_i, int Ly_i, int Lz_i,
                               double *p_o, int Lx_o, int Ly_o, int Lz_o,
                               double Sx, double Sy, double Sz)
{
    for (int k = 0; k < Lz_o; k++)
    {
        double z_o = (k - 0.5) * Sz / Lz_o;
        int k0_i = std::clamp((k * Lz_i) / Lz_o, 0, Lz_i - 2);
        int k1_i = std::clamp((k * Lz_i) / Lz_o + 1, 1, Lz_i - 1);
        double z0_i = (k0_i - 0.5) * Sz / Lz_i;
        double z1_i = (k1_i - 0.5) * Sz / Lz_i;

        for (int j = 0; j < Ly_o; j++)
        {
            double y_o = (j - 0.5) * Sy / Ly_o;
            int j0_i = std::clamp((j * Ly_i) / Ly_o, 0, Ly_i - 2);
            int j1_i = std::clamp((j * Ly_i) / Ly_o + 1, 1, Ly_i - 1);
            double y0_i = (j0_i - 0.5) * Sy / Ly_i;
            double y1_i = (j1_i - 0.5) * Sy / Ly_i;

            for (int i = 0; i < Lx_o; i++)
            {
                double x_o = (i - 0.5) * Sx / Lx_o;
                int i0_i = std::clamp((i * Lx_i) / Lx_o, 0, Lx_i - 2);
                int i1_i = std::clamp((i * Lx_i) / Lx_o + 1, 1, Lx_i - 1);
                double x0_i = (i0_i - 0.5) * Sx / Lx_i;
                double x1_i = (i1_i - 0.5) * Sx / Lx_i;

                p_o[(k * Ly_o + j) * Lx_o + i] = Lerp(z_o, z0_i, z1_i,
                                                      Lerp(y_o, y0_i, y1_i,
                                                           Lerp(x_o, x0_i, x1_i, p_i[(k0_i * Ly_i + j0_i) * Lx_i + i0_i], p_i[(k0_i * Ly_i + j0_i) * Lx_i + i1_i]),
                                                           Lerp(x_o, x0_i, x1_i, p_i[(k0_i * Ly_i + j1_i) * Lx_i + i0_i], p_i[(k0_i * Ly_i + j1_i) * Lx_i + i1_i])),
                                                      Lerp(y_o, y0_i, y1_i,
                                                           Lerp(x_o, x0_i, x1_i, p_i[(k1_i * Ly_i + j0_i) * Lx_i + i0_i], p_i[(k1_i * Ly_i + j0_i) * Lx_i + i1_i]),
                                                           Lerp(x_o, x0_i, x1_i, p_i[(k1_i * Ly_i + j1_i) * Lx_i + i0_i], p_i[(k1_i * Ly_i + j1_i) * Lx_i + i1_i])));
            }
        }
    }
}