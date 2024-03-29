#include "../include/interpolationGPU.hpp"

__device__ inline double lerp(double x, double x0, double x1, double v0, double v1)
{
    if (abs(x0 - x1) < 1.0e-8)
        return (v0 + v1) / 2;
    return (x1 - x) / (x1 - x0) * v0 + (x - x0) / (x1 - x0) * v1;
}

__global__ void CUDA_Rescale(double *p_i, int Lx_i, double *p_o, int Lx_o, double Sx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= Lx_o)
        return;

    double x_o = (i + 0.5) * Sx / Lx_o;
    int i0_i = int(floor(x_o * Lx_i / Sx - 0.5));
    int i1_i = int(ceil(x_o * Lx_i / Sx - 0.5));
    // Min
    if (i0_i < 0)
    {
        i0_i = 0;
        i1_i = 1;
    }
    // Max
    if (i1_i >= Lx_i)
    {
        i0_i = Lx_i - 2;
        i1_i = Lx_i - 1;
    }
    double x0_i = (i0_i + 0.5) * Sx / Lx_i;
    double x1_i = (i1_i + 0.5) * Sx / Lx_i;
    p_o[i] = lerp(x_o, x0_i, x1_i, p_i[i0_i], p_i[i1_i]);
}
__global__ void CUDA_Rescale(double *p_i, int Lx_i, int Ly_i, double *p_o, int Lx_o, int Ly_o, double Sx, double Sy)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= Lx_o || j >= Ly_o)
        return;

    double x_o = (i + 0.5) * Sx / Lx_o;
    int i0_i = int(floor(x_o * Lx_i / Sx - 0.5));
    int i1_i = int(ceil(x_o * Lx_i / Sx - 0.5));
    // Min
    if (i0_i < 0)
    {
        i0_i = 0;
        i1_i = 1;
    }
    // Max
    if (i1_i >= Lx_i)
    {
        i0_i = Lx_i - 2;
        i1_i = Lx_i - 1;
    }
    double x0_i = (i0_i + 0.5) * Sx / Lx_i;
    double x1_i = (i1_i + 0.5) * Sx / Lx_i;

    double y_o = (j + 0.5) * Sy / Ly_o;
    int j0_i = int(floor(y_o * Ly_i / Sy - 0.5));
    int j1_i = int(ceil(y_o * Ly_i / Sy - 0.5));
    // Min
    if (j0_i < 0)
    {
        j0_i = 0;
        j1_i = 1;
    }
    // Max
    if (j1_i >= Ly_i)
    {
        j0_i = Ly_i - 2;
        j1_i = Ly_i - 1;
    }
    double y0_i = (j0_i + 0.5) * Sy / Ly_i;
    double y1_i = (j1_i + 0.5) * Sy / Ly_i;

    p_o[j * Lx_o + i] = lerp(y_o, y0_i, y1_i,
                             lerp(x_o, x0_i, x1_i, p_i[j0_i * Lx_i + i0_i], p_i[j0_i * Lx_i + i1_i]),
                             lerp(x_o, x0_i, x1_i, p_i[j1_i * Lx_i + i0_i], p_i[j1_i * Lx_i + i1_i]));
}
__global__ void CUDA_Rescale(double *p_i, int Lx_i, int Ly_i, int Lz_i, double *p_o, int Lx_o, int Ly_o, int Lz_o, double Sx, double Sy, double Sz)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= Lx_o || j >= Ly_o || k >= Lz_o)
        return;

    double x_o = (i + 0.5) * Sx / Lx_o;
    int i0_i = int(floor(x_o * Lx_i / Sx - 0.5));
    int i1_i = int(ceil(x_o * Lx_i / Sx - 0.5));
    // Min
    if (i0_i < 0)
    {
        i0_i = 0;
        i1_i = 1;
    }
    // Max
    if (i1_i >= Lx_i)
    {
        i0_i = Lx_i - 2;
        i1_i = Lx_i - 1;
    }
    double x0_i = (i0_i + 0.5) * Sx / Lx_i;
    double x1_i = (i1_i + 0.5) * Sx / Lx_i;

    double y_o = (j - 0.5) * Sy / Ly_o;
    int j0_i = int(floor(y_o * Ly_i / Sy - 0.5));
    int j1_i = int(ceil(y_o * Ly_i / Sy - 0.5));
    // Min
    if (j0_i < 0)
    {
        j0_i = 0;
        j1_i = 1;
    }
    // Max
    if (j1_i >= Ly_i)
    {
        j0_i = Ly_i - 2;
        j1_i = Ly_i - 1;
    }
    double y0_i = (j0_i + 0.5) * Sy / Ly_i;
    double y1_i = (j1_i + 0.5) * Sy / Ly_i;

    double z_o = (k + 0.5) * Sz / Lz_o;
    int k0_i = int(floor(z_o * Lz_i / Sz - 0.5));
    int k1_i = int(ceil(z_o * Lz_i / Sz - 0.5));
    // Min
    if (k0_i < 0)
    {
        k0_i = 0;
        k1_i = 1;
    }
    // Max
    if (k1_i >= Lz_i)
    {
        k0_i = Lz_i - 2;
        k1_i = Lz_i - 1;
    }
    double z0_i = (k0_i + 0.5) * Sz / Lz_i;
    double z1_i = (k1_i + 0.5) * Sz / Lz_i;

    p_o[(k * Ly_o + j) * Lx_o + i] = lerp(z_o, z0_i, z1_i,
                                          lerp(y_o, y0_i, y1_i,
                                               lerp(x_o, x0_i, x1_i, p_i[(k0_i * Ly_i + j0_i) * Lx_i + i0_i], p_i[(k0_i * Ly_i + j0_i) * Lx_i + i1_i]),
                                               lerp(x_o, x0_i, x1_i, p_i[(k0_i * Ly_i + j1_i) * Lx_i + i0_i], p_i[(k0_i * Ly_i + j1_i) * Lx_i + i1_i])),
                                          lerp(y_o, y0_i, y1_i,
                                               lerp(x_o, x0_i, x1_i, p_i[(k1_i * Ly_i + j0_i) * Lx_i + i0_i], p_i[(k1_i * Ly_i + j0_i) * Lx_i + i1_i]),
                                               lerp(x_o, x0_i, x1_i, p_i[(k1_i * Ly_i + j1_i) * Lx_i + i0_i], p_i[(k1_i * Ly_i + j1_i) * Lx_i + i1_i])));
}

void InterpolationGPU::Rescale(double *p_i, int Lx_i,
                               double *p_o, int Lx_o,
                               double Sx)
{
    dim3 T(1024);
    dim3 B(CEIL(Lx_o, T.x));
    CUDA_Rescale<<<B, T, 0, cudaStreamDefault>>>(p_i, Lx_i, p_o, Lx_o, Sx);
}
void InterpolationGPU::Rescale(double *p_i, int Lx_i, int Ly_i,
                               double *p_o, int Lx_o, int Ly_o,
                               double Sx, double Sy)
{
    dim3 T(32, 32);
    dim3 B(CEIL(Lx_o, T.x), CEIL(Ly_o, T.y));
    CUDA_Rescale<<<B, T, 0, cudaStreamDefault>>>(p_i, Lx_i, Ly_i, p_o, Lx_o, Ly_o, Sx, Sy);
}
void InterpolationGPU::Rescale(double *p_i, int Lx_i, int Ly_i, int Lz_i,
                               double *p_o, int Lx_o, int Ly_o, int Lz_o,
                               double Sx, double Sy, double Sz)
{
    dim3 T(16, 16, 4);
    dim3 B(CEIL(Lx_o, T.x), CEIL(Ly_o, T.y), CEIL(Lz_o, T.z));
    CUDA_Rescale<<<B, T, 0, cudaStreamDefault>>>(p_i, Lx_i, Ly_i, Lz_i, p_o, Lx_o, Ly_o, Lz_o, Sx, Sy, Sz);
}
