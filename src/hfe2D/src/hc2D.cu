#include "../include/hc2D.hpp"

__host__ __device__ inline double HC2D::C(double x, double y)
{
    return x + y;
}
__host__ __device__ inline double HC2D::K(double x, double y)
{
    return x + y;
}

void HC2D::CPU::EvolutionJacobianMatrix(double *pmTT_o, double *pmTQ_o, double *pmQT_o, double *pmQQ_o, HCParms &parms)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double amp = parms.amp;
    double h = parms.h;

    // Difusion Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double KT = K((i + 0.5) * dx, (j + 0.5) * dy);
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            double aux = 0.0;
            if (i != 0)
            {
                aux += KT / (CT * dx * dx);
                pmTT_o[(index - 1) * L + index] = KT / (CT * dx * dx);
            }
            if (i != Lx - 1)
            {
                aux += KT / (CT * dx * dx);
                pmTT_o[(index + 1) * L + index] = KT / (CT * dx * dx);
            }
            if (j != 0)
            {
                aux += KT / (CT * dy * dy);
                pmTT_o[(index - Lx) * L + index] = KT / (CT * dy * dy);
            }
            if (j != Ly - 1)
            {
                aux += KT / (CT * dy * dy);
                pmTT_o[(index + Lx) * L + index] = KT / (CT * dy * dy);
            }
            pmTT_o[index * L + index] = -aux - h / dz;
        }
    }

    // Heat Flux Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pmQT_o[index * L + index] = amp / (dz * CT);
        }
    }
}

void HC2D::CPU::EvolutionControlMatrix(double *pmUT_o, double *pmUQ_o, HCParms &parms)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double h = parms.h;

    // Control Input Temperature Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pmUT_o[index] = h / (dz * CT);
        }
    }
    // Control Input Heat Flux Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            index = j * Lx + i;
            pmUQ_o[index] = 0.0;
        }
    }
}

void HC2D::CPU::EvaluationMatrix(double *pmTT_o, double *pmQT_o, HCParms &parms)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    double dx = parms.dx;
    double dy = parms.dy;
    double dz = parms.dz;

    // Surface Temperature
    for (int j = 0; j < Ly; j++)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double KT = K((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pmTT_o[index * Lxy + index] = 1;
#if ILSA == 1
            pmQT_o[index * Lxy + index] = -dz * parms.amp / (6 * KT);
#endif
        }
    }
}

void HC2D::GPU::EvolutionJacobianMatrix(double *pmTT_o, double *pmTQ_o, double *pmQT_o, double *pmQQ_o, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;

    double *pm, *pm_o, *pmTT, *pmTQ, *pmQT, *pmQQ;

    int stride11 = pmTQ_o - pmTT_o;
    int stride22 = pmQQ_o - pmTT_o;

    pm = (double *)malloc(sizeof(double) * 4 * Lxy * Lxy);
    for (int i = 0; i < 4 * Lxy * Lxy; i++)
    {
        pm[i] = 0.0;
    }

    pmTT = pm + std::max(-stride22, 0);
    pmTQ = pm + std::max(-stride22, 0) + std::max(stride11, 0);
    pmQT = pm + std::max(stride22, 0) + std::max(-stride11, 0);
    pmQQ = pm + std::max(stride22, 0);

    HC2D::CPU::EvolutionJacobianMatrix(pmTT, pmTQ, pmQT, pmQQ, parms);

    pm_o = std::min(pmTT_o, pmQQ_o);
    cudaMemcpy(pm_o, pm, 4 * sizeof(double) * Lxy * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}

void HC2D::GPU::EvolutionControlMatrix(double *pmUT_o, double *pmUQ_o, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    double *pm, *pm_o, *pmTT, *pmQT;
    int stride12 = pmUT_o - pmUQ_o;
    
    pm = (double *)malloc(sizeof(double) * 2 * Lxy);
    for (int i = 0; i < 2 * Lxy; i++)
    {
        pm[i] = 0.0;
    }
    pmTT = pm + std::max(-stride12, 0);
    pmQT = pm + std::max(stride12, 0);

    HC2D::CPU::EvolutionControlMatrix(pmTT, pmQT, parms);

    pm_o = std::min(pmUT_o, pmUQ_o);
    cudaMemcpy(pm_o, pm, 2 * sizeof(double) * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}

void HC2D::GPU::EvaluationMatrix(double *pmTT_o, double *pmQT_o, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;

    double *pm, *pm_o, *pmTT, *pmQT;

    int stride12 = pmQT_o - pmTT_o;

    pm = (double *)malloc(sizeof(double) * 2 * Lxy * Lxy);
    for (int i = 0; i < 2 * Lxy * Lxy; i++)
    {
        pm[i] = 0.0;
    }

    pmTT = pm + std::max(-stride12, 0);
    pmQT = pm + std::max(stride12, 0);

    HC2D::CPU::EvaluationMatrix(pmTT, pmQT, parms);

    pm_o = std::min(pmTT_o, pmQT_o);
    cudaMemcpy(pm_o, pm, 2 * sizeof(double) * Lxy * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}