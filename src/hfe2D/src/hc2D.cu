#include "../include/hc2D.hpp"

__host__ __device__ inline double HC2D::C(double x, double y)
{
    return 385 * 8960;
}
__host__ __device__ inline double HC2D::K(double x, double y)
{
    return 400;
}

void HC2D::validate(HCParms &parms)
{
    if (!(refparms == parms))
    {
        refparms = parms;
        if (AI.host())
        {
            AI.free();
        }
        int Lxy = parms.Lx * parms.Ly;
        int Lu = 1 + 2 * (parms.Lx + parms.Ly);
        int L = Lxy + Lxy;
        int L2 = L * L;
        AI.alloc(L2);
        BE.alloc(L2);
        CE.alloc(L * Lu);
        ATA.alloc(L2);
        JX.alloc(L2);
        JU.alloc(L * Lu);
        isValid = false;
    }
}

void HC2D::CPU::ImplicitScheme(HCParms &parms, int strideTQ, int strideAC)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1 + 2 * (parms.Lx + parms.Ly);
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double dt = parms.dt;
    double c = parms.Sz;
    double amp = parms.amp;
    double h = parms.h;
    double gamma = parms.gamma;

    double *pAI = AI.host();
    double *pBE = BE.host();
    double *pCE = CE.host();
    double *pATA = ATA.host();
    double *paux = (double *)malloc(sizeof(double) * L * L);
    double *pTT, *pTQ, *pQT, *pQQ, *pTaT, *pTaQ, *pTcT, *pTcQ;
    MathCPU::Zero(pAI, L * L);
    MathCPU::Identity(pBE, L, L);
    MathCPU::Zero(pCE, L * Lu);
    MathCPU::Zero(pATA, L * L);
    MathCPU::Zero(paux, L * L);
    pTT = pAI + std::max(0, -strideTQ) * (L + 1);
    pQQ = pAI + std::max(0, strideTQ) * (L + 1);
    pTQ = pTT + strideTQ;
    pQT = pQQ - strideTQ;
    pTaT = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, -strideTQ);
    pTaQ = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, strideTQ);
    pTcT = pCE + std::max(0, strideAC) * L + std::max(0, -strideTQ);
    pTcQ = pCE + std::max(0, strideAC) * L + std::max(0, strideTQ);

    double *JXh = JX.host();
    double *JUh = JU.host();
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
                aux += dt * KT / (CT * dx * dx);
                pTT[(index - 1) * L + index] = -dt * KT / (CT * dx * dx);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
                pTT[(index + 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            }
            if (i != Lx - 1)
            {
                aux += dt * KT / (CT * dx * dx);
                pTT[(index + 1) * L + index] = -dt * KT / (CT * dx * dx);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
                pTT[(index - 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            }
            if (j != 0)
            {
                aux += dt * KT / (CT * dy * dy);
                pTT[(index - Lx) * L + index] = -dt * KT / (CT * dy * dy);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
                pTT[(index + Lx) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            }
            if (j != Ly - 1)
            {
                aux += dt * KT / (CT * dy * dy);
                pTT[(index + Lx) * L + index] = -dt * KT / (CT * dy * dy);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
                pTT[(index - Lx) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            }
            pTT[index * L + index] = 1.0 + aux + dt * h / (c * CT);
        }
    }

    // Heat Flux Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pQT[index * L + index] = -dt * amp / (c * CT);
        }
    }
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pQQ[index * L + index] = 1.0;
        }
    }

    // Ambient Temperature Temperature Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pTaT[index] = h / (c * CT);
        }
    }
    // Contour Temperature Temperature Contribution
    for (int j = 0; j < Ly; ++j)
    {
        double KT0 = K((0.0) * dx, (j + 0.5) * dy);
        double KT1 = K((Lx)*dx, (j + 0.5) * dy);
        double CT0 = C((0.0) * dx, (j + 0.5) * dy);
        double CT1 = C((Lx)*dx, (j + 0.5) * dy);
        int index0 = j * Lx + 0;
        int index1 = j * Lx + Lx - 1;
        pTcT[j * L + index0] = -dt * gamma * KT0 / (CT0 * dx * dx);
        pTcT[(Ly + j) * L + index1] = -dt * gamma * KT1 / (CT1 * dx * dx);
    }
    for (int i = 0; i < Lx; ++i)
    {
        double KT0 = K((i + 0.5) * dx, (0.0) * dy);
        double KT1 = K((i + 0.5) * dx, (Ly)*dy);
        double CT0 = C((i + 0.5) * dx, (0.0) * dy);
        double CT1 = C((i + 0.5) * dx, (Ly)*dy);
        int index0 = 0 * Lx + i;
        int index1 = (Ly - 1) * Lx + i;
        pTcT[(2 * Ly + i) * L + index0] = -dt * gamma * KT0 / (CT0 * dy * dy);
        pTcT[(2 * Ly + Lx + i) * L + index1] = -dt * gamma * KT1 / (CT1 * dy * dy);
    }

    MathCPU::MatMulTN(0.0, pATA, 1.0, pAI, pAI, L, L, L);
    MathCPU::MatMulTN(0.0, paux, 1.0, pAI, pBE, L, L, L);
    // Solve JX = (A^T * A)^-1 * A^T * B
    MathCPU::CholeskySolver(JXh, pATA, paux, L, L, L);

    MathCPU::MatMulTN(0.0, paux, 1.0, pAI, pCE, L, L, Lu);
    // Solve JU = (A^T * A)^-1 * A^T * C
    MathCPU::CholeskySolver(JUh, pATA, paux, L, L, Lu);
    free(paux);
}

void HC2D::CPU::ExplicitScheme(HCParms &parms, int strideTQ, int strideAC)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1 + 2 * (parms.Lx + parms.Ly);
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double dt = parms.dt;
    double c = parms.Sz;
    double amp = parms.amp;
    double h = parms.h;
    double gamma = parms.gamma;

    double *pAI = AI.host();
    double *pBE = BE.host();
    double *pCE = CE.host();
    double *pATA = ATA.host();
    double *paux = (double *)malloc(sizeof(double) * L * L);
    double *pTT, *pTQ, *pQT, *pQQ, *pTaT, *pTaQ, *pTcT, *pTcQ;
    MathCPU::Identity(pAI, L, L);
    MathCPU::Zero(pBE, L * L);
    MathCPU::Zero(pCE, L * Lu);
    MathCPU::Zero(pATA, L * L);
    MathCPU::Zero(paux, L * L);
    pTT = pAI + std::max(0, -strideTQ) * (L + 1);
    pQQ = pAI + std::max(0, strideTQ) * (L + 1);
    pTQ = pTT + strideTQ;
    pQT = pQQ - strideTQ;
    pTaT = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, -strideTQ);
    pTaQ = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, strideTQ);
    pTcT = pCE + std::max(0, strideAC) * L + std::max(0, -strideTQ);
    pTcQ = pCE + std::max(0, strideAC) * L + std::max(0, strideTQ);

    double *JXh = JX.host();
    double *JUh = JU.host();
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
                aux += dt * KT / (CT * dx * dx);
                pTT[(index - 1) * L + index] = dt * KT / (CT * dx * dx);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
                pTT[(index + 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            }
            if (i != Lx - 1)
            {
                aux += dt * KT / (CT * dx * dx);
                pTT[(index + 1) * L + index] = dt * KT / (CT * dx * dx);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
                pTT[(index - 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            }
            if (j != 0)
            {
                aux += dt * KT / (CT * dy * dy);
                pTT[(index - Lx) * L + index] = dt * KT / (CT * dy * dy);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
                pTT[(index + Lx) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            }
            if (j != Ly - 1)
            {
                aux += dt * KT / (CT * dy * dy);
                pTT[(index + Lx) * L + index] = dt * KT / (CT * dy * dy);
            }
            else
            {
                aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
                pTT[(index - Lx) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            }
            pTT[index * L + index] = 1.0 - aux - dt * h / (c * CT);
        }
    }

    // Heat Flux Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pQT[index * L + index] = dt * amp / (c * CT);
        }
    }
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pQQ[index * L + index] = 1.0;
        }
    }

    // Ambient Temperature Temperature Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double CT = C((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pTaT[index] = h / (c * CT);
        }
    }
    // Contour Temperature Temperature Contribution
    for (int j = 0; j < Ly; ++j)
    {
        double KT0 = K((0.0) * dx, (j + 0.5) * dy);
        double KT1 = K((Lx)*dx, (j + 0.5) * dy);
        double CT0 = C((0.0) * dx, (j + 0.5) * dy);
        double CT1 = C((Lx)*dx, (j + 0.5) * dy);
        int index0 = j * Lx + 0;
        int index1 = j * Lx + Lx - 1;
        pTcT[j * L + index0] = -dt * gamma * KT0 / (CT0 * dx * dx);
        pTcT[(Ly + j) * L + index1] = -dt * gamma * KT1 / (CT1 * dx * dx);
    }
    for (int i = 0; i < Lx; ++i)
    {
        double KT0 = K((i + 0.5) * dx, (0.0) * dy);
        double KT1 = K((i + 0.5) * dx, (Ly)*dy);
        double CT0 = C((i + 0.5) * dx, (0.0) * dy);
        double CT1 = C((i + 0.5) * dx, (Ly)*dy);
        int index0 = 0 * Lx + i;
        int index1 = (Ly - 1) * Lx + i;
        pTcT[(2 * Ly + i) * L + index0] = -dt * gamma * KT0 / (CT0 * dy * dy);
        pTcT[(2 * Ly + Lx + i) * L + index1] = -dt * gamma * KT1 / (CT1 * dy * dy);
    }

    // Solve JX = (A^T * A)^-1 * A^T * B
    MathCPU::Copy(JXh, pBE, L * L);

    // Solve JU = (A^T * A)^-1 * A^T * C
    MathCPU::Copy(JUh, pCE, L * Lu);
}

void HC2D::CPU::EvolutionMatrix(HCParms &parms, double *pmXX_o, double *pmUX_o, int strideTQ, int strideAC)
{
    validate(parms);
    if (!isValid)
    {
#if IMPLICIT_SCHEME == 1
        ImplicitScheme(parms, strideTQ, strideAC);
#else
        ExplicitScheme(parms, strideTQ, strideAC);
#endif
        isValid = true;
    }
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1;
    int L = Lxy + Lxy;
    MathCPU::Copy(pmXX_o, JX.host(), L * L);
    MathCPU::Copy(pmUX_o, JU.host(), L * Lu);
}

void HC2D::CPU::EvaluationMatrix(HCParms &parms, double *pmH_o, int strideTQ)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    double dx = parms.dx;
    double dy = parms.dy;
    double c = parms.Sz;
    double amp = parms.amp;

    double *pmTT, *pmQT;
    pmTT = pmH_o + std::max(-strideTQ, 0) * Lxy;
    pmQT = pmH_o + std::max(strideTQ, 0) * Lxy;
    // Surface Temperature
    for (int j = 0; j < Ly; j++)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double KT = K((i + 0.5) * dx, (j + 0.5) * dy);
            index = j * Lx + i;
            pmTT[index * Lxy + index] = 1.0;
#if ILSA == 1
            pmQT[index * Lxy + index] = -c * amp / (6.0 * KT);
#endif
        }
    }
}

__global__ void ImplicitScheme_A(double *pmTT, double *pmTQ, double *pmQT, double *pmQQ, int Lx, int Ly, double dx, double dy, double dt, double c, double amp, double h, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int Lxy = Lx * Ly;
    int L = Lxy + Lxy;
    int index = j * Lx + i;
    if (i < Lx && j < Ly)
    {
        double KT = HC2D::K((i + 0.5) * dx, (j + 0.5) * dy);
        double CT = HC2D::C((i + 0.5) * dx, (j + 0.5) * dy);
        double aux = 0.0;
        if (i != 0)
        {
            aux += dt * KT / (CT * dx * dx);
            pmTT[(index - 1) * L + index] = -dt * KT / (CT * dx * dx);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
        }
        if (i != Lx - 1)
        {
            aux += dt * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] = -dt * KT / (CT * dx * dx);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
        }
        if (j != 0)
        {
            aux += dt * KT / (CT * dy * dy);
            pmTT[(index - Lx) * L + index] = -dt * KT / (CT * dy * dy);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            pmTT[(index + 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
        }
        if (j != Ly - 1)
        {
            aux += dt * KT / (CT * dy * dy);
            pmTT[(index + Lx) * L + index] = -dt * KT / (CT * dy * dy);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            pmTT[(index + 1) * L + index] += -(1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
        }
        pmTT[index * L + index] = 1.0 + aux + dt * h / (c * CT);
        pmQT[index * L + index] = -dt * amp / (c * CT);
        pmQQ[index * L + index] = 1.0;
    }
}

__global__ void ImplicitScheme_C(double *pmTaT, double *pmTaQ, double *pmTcT, double *pmTcQ, int Lx, int Ly, double dx, double dy, double dt, double c, double amp, double h, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int Lxy = Lx * Ly;
    int L = Lxy + Lxy;
    int index = j * Lx + i;
    if (i < Lx && j < Ly)
    {
        double KT = HC2D::K((i + 0.5) * dx, (j + 0.5) * dy);
        double CT = HC2D::C((i + 0.5) * dx, (j + 0.5) * dy);
        pmTaT[index] = dt * h / (c * CT);
        if (i == 0)
        {
            for (int k = 0; k < Ly; k++)
            {
                pmTcT[k * L + index] = -dt * gamma * KT / (CT * dx * dx);
            }
        }
        if (i == Lx - 1)
        {
            for (int k = 0; k < Ly; k++)
            {
                pmTcT[(k + Ly) * L + index] = -dt * gamma * KT / (CT * dx * dx);
            }
        }
        if (j == 0)
        {
            for (int k = 0; k < Lx; k++)
            {
                pmTcT[(k + 2 * Ly) * L + index] = -dt * gamma * KT / (CT * dy * dy);
            }
        }
        if (j == Ly - 1)
        {
            for (int k = 0; k < Lx; k++)
            {
                pmTcT[(k + Lx + 2 * Ly) * L + index] = -dt * gamma * KT / (CT * dy * dy);
            }
        }
    }
}

__global__ void ExplicitScheme_B(double *pmTT, double *pmTQ, double *pmQT, double *pmQQ, int Lx, int Ly, double dx, double dy, double dt, double c, double amp, double h, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int Lxy = Lx * Ly;
    int L = Lxy + Lxy;
    int index = j * Lx + i;
    if (i < Lx && j < Ly)
    {
        double KT = HC2D::K((i + 0.5) * dx, (j + 0.5) * dy);
        double CT = HC2D::C((i + 0.5) * dx, (j + 0.5) * dy);
        double aux = 0.0;
        if (i != 0)
        {
            aux += dt * KT / (CT * dx * dx);
            pmTT[(index - 1) * L + index] = dt * KT / (CT * dx * dx);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
        }
        if (i != Lx - 1)
        {
            aux += dt * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] = dt * KT / (CT * dx * dx);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
            pmTT[(index + 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dx * dx);
        }
        if (j != 0)
        {
            aux += dt * KT / (CT * dy * dy);
            pmTT[(index - Lx) * L + index] = dt * KT / (CT * dy * dy);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            pmTT[(index + 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
        }
        if (j != Ly - 1)
        {
            aux += dt * KT / (CT * dy * dy);
            pmTT[(index + Lx) * L + index] = dt * KT / (CT * dy * dy);
        }
        else
        {
            aux += (3.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
            pmTT[(index + 1) * L + index] += (1.0 / 2.0) * dt * gamma * KT / (CT * dy * dy);
        }
        pmTT[index * L + index] = 1.0 - aux - dt * h / (c * CT);
        pmQT[index * L + index] = dt * amp / (c * CT);
        pmQQ[index * L + index] = 1.0;
    }
}

__global__ void ExplicitScheme_C(double *pmTaT, double *pmTaQ, double *pmTcT, double *pmTcQ, int Lx, int Ly, double dx, double dy, double dt, double c, double amp, double h, double gamma)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int Lxy = Lx * Ly;
    int L = Lxy + Lxy;
    int index = j * Lx + i;
    if (i < Lx && j < Ly)
    {
        double KT = HC2D::K((i + 0.5) * dx, (j + 0.5) * dy);
        double CT = HC2D::C((i + 0.5) * dx, (j + 0.5) * dy);
        pmTaT[index] = dt * h / (c * CT);
        if (i == 0)
        {
            for (int k = 0; k < Ly; k++)
            {
                pmTcT[k * L + index] = -dt * gamma * KT / (CT * dx * dx);
            }
        }
        if (i == Lx - 1)
        {
            for (int k = 0; k < Ly; k++)
            {
                pmTcT[(k + Ly) * L + index] = -dt * gamma * KT / (CT * dx * dx);
            }
        }
        if (j == 0)
        {
            for (int k = 0; k < Lx; k++)
            {
                pmTcT[(k + 2 * Ly) * L + index] = -dt * gamma * KT / (CT * dy * dy);
            }
        }
        if (j == Ly - 1)
        {
            for (int k = 0; k < Lx; k++)
            {
                pmTcT[(k + Lx + 2 * Ly) * L + index] = -dt * gamma * KT / (CT * dy * dy);
            }
        }
    }
}

void HC2D::GPU::ImplicitScheme(HCParms &parms, int strideTQ, int strideAC)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1;
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double dt = parms.dt;
    double c = parms.Sz;
    double amp = parms.amp;
    double h = parms.h;
    double gamma = parms.gamma;

    double *pAI = AI.dev();
    double *pBE = BE.dev();
    double *pCE = CE.dev();
    double *pATA = ATA.dev();
    double *paux = NULL;
    cudaMalloc(&paux, sizeof(double) * L * L);
    double *pTT, *pTQ, *pQT, *pQQ, *pTaT, *pTaQ, *pTcT, *pTcQ;
    MathGPU::Zero(pAI, L * L);
    MathGPU::Identity(pBE, L, L);
    MathGPU::Zero(pCE, L * Lu);
    MathGPU::Zero(pATA, L * L);
    MathGPU::Zero(paux, L * L);
    pTT = pAI + std::max(0, -strideTQ) * (L + 1);
    pQQ = pAI + std::max(0, strideTQ) * (L + 1);
    pTQ = pTT + strideTQ;
    pQT = pQQ - strideTQ;
    pTaT = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, -strideTQ);
    pTaQ = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, strideTQ);
    pTcT = pCE + std::max(0, strideAC) * L + std::max(0, -strideTQ);
    pTcQ = pCE + std::max(0, strideAC) * L + std::max(0, strideTQ);

    double *JXh = JX.dev();
    double *JUh = JU.dev();

    dim3 T(16, 16);
    dim3 B((L + T.x - 1) / T.x, (L + T.y - 1) / T.y);
    ImplicitScheme_A<<<B, T>>>(pTT, pTQ, pQT, pQQ, Lx, Ly, dx, dy, dt, c, amp, h, gamma);
    MathGPU::Identity(pBE, L, L);
    ImplicitScheme_C<<<B, T>>>(pTaT, pTaQ, pTcT, pTcQ, Lx, Ly, dx, dy, dt, c, amp, h, gamma);

    MathGPU::MatMulTN(0.0, pATA, 1.0, pAI, pAI, L, L, L);
    MathGPU::MatMulTN(0.0, paux, 1.0, pAI, pBE, L, L, L);
    // Solve JX = (A^T * A)^-1 * A^T * B
    MathGPU::CholeskySolver(JXh, pATA, paux, L, L, L);

    MathGPU::MatMulTN(0.0, paux, 1.0, pAI, pCE, L, L, Lu);
    // Solve JU = (A^T * A)^-1 * A^T * C
    MathGPU::CholeskySolver(JUh, pATA, paux, L, L, Lu);
    cudaFree(paux);
}

void HC2D::GPU::ExplicitScheme(HCParms &parms, int strideTQ, int strideAC)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1;
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double dt = parms.dt;
    double c = parms.Sz;
    double amp = parms.amp;
    double h = parms.h;
    double gamma = parms.gamma;

    double *pAI = AI.dev();
    double *pBE = BE.dev();
    double *pCE = CE.dev();
    double *pATA = ATA.dev();
    double *paux = NULL;
    cudaMalloc(&paux, sizeof(double) * L * L);
    double *pTT, *pTQ, *pQT, *pQQ, *pTaT, *pTaQ, *pTcT, *pTcQ;
    MathGPU::Identity(pAI, L, L);
    MathGPU::Zero(pBE, L * L);
    MathGPU::Zero(pCE, L * Lu);
    MathGPU::Zero(pATA, L * L);
    MathGPU::Zero(paux, L * L);
    pTT = pAI + std::max(0, -strideTQ) * (L + 1);
    pQQ = pAI + std::max(0, strideTQ) * (L + 1);
    pTQ = pTT + strideTQ;
    pQT = pQQ - strideTQ;
    pTaT = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, -strideTQ);
    pTaQ = pCE + std::max(0, -strideAC) * (2 * (parms.Lx + parms.Ly)) * L + std::max(0, strideTQ);
    pTcT = pCE + std::max(0, strideAC) * L + std::max(0, -strideTQ);
    pTcQ = pCE + std::max(0, strideAC) * L + std::max(0, strideTQ);

    double *JXh = JX.dev();
    double *JUh = JU.dev();

    dim3 T(16, 16);
    dim3 B((L + T.x - 1) / T.x, (L + T.y - 1) / T.y);
    ImplicitScheme_A<<<B, T>>>(pTT, pTQ, pQT, pQQ, Lx, Ly, dx, dy, dt, c, amp, h, gamma);
    MathGPU::Identity(pBE, L, L);
    ImplicitScheme_C<<<B, T>>>(pTaT, pTaQ, pTcT, pTcQ, Lx, Ly, dx, dy, dt, c, amp, h, gamma);

    // Solve JX = (A^T * A)^-1 * A^T * B
    MathGPU::Copy(JXh, pBE, L * L);

    // Solve JU = (A^T * A)^-1 * A^T * C
    MathGPU::Copy(JUh, pCE, L * Lu);
}

void HC2D::GPU::EvolutionMatrix(HCParms &parms, double *pmXX_o, double *pmUX_o, int strideTQ, int strideAC)
{
    validate(parms);
    if (!isValid)
    {
#if IMPLICIT_SCHEME == 1
        ImplicitScheme(parms, strideTQ, strideAC);
#else
        ExplicitScheme(parms, strideTQ, strideAC);
#endif
        std::cout << "Saving matrices for Evolution" << std::endl;
        isValid = true;
    }
    int Lxy = parms.Lx * parms.Ly;
    int Lu = 1;
    int L = Lxy + Lxy;
    MathGPU::Copy(pmXX_o, JX.dev(), L * L);
    MathGPU::Copy(pmUX_o, JU.dev(), L * Lu);
}

void HC2D::GPU::EvaluationMatrix(HCParms &parms, double *pmH_o, int strideTQ)
{
    int Lxy = parms.Lx * parms.Ly;

    double *pm;

    pm = (double *)malloc(sizeof(double) * 2 * Lxy * Lxy);
    for (int i = 0; i < 2 * Lxy * Lxy; i++)
    {
        pm[i] = 0.0;
    }

    HC2D::CPU::EvaluationMatrix(parms, pm, strideTQ);

    cudaMemcpy(pmH_o, pm, 2 * sizeof(double) * Lxy * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}