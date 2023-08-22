#include "../include/hfe.hpp"

void HFE::EvolveCPU(Data *pstate)
{
    double *pinstance = pstate->GetInstances().host();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int s = 0; s < Lsigma; ++s)
    {
#if FORWARD_METHOD == 0
        HC::CPU::Euler(pinstance + offsetT + Lstate * s, pinstance + offsetQ + Lstate * s, workspace, parms);
#elif FORWARD_METHOD == 1
        HC::CPU::RK4(pinstance + offsetT + Lstate * s, pinstance + offsetQ + Lstate * s, workspace, parms);
#elif FORWARD_METHOD == 2
        HC::CPU::RKF45(pinstance + offsetT + Lstate * s, pinstance + offsetQ + Lstate * s, workspace, parms);
#endif
    }
}
void HFE::EvolveGPU(Data *pstate)
{
    double *pinstance = pstate->GetInstances().dev();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int i = 0; i < Lsigma; ++i)
    {
        double *T = pinstance + Lstate * i + offsetT;
        double *Q = pinstance + Lstate * i + offsetQ;
    }
}
void HFE::EvaluateCPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetInstances().host();
    double *pminstance = pmeasure->GetInstances().host();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int Lmeasure = pmeasure->GetMeasureLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    for (int i = 0; i < Lsigma; ++i)
    {
        double *T = psinstance + Lstate * i + offsetT;
        double *Q = psinstance + Lstate * i + offsetQ;
        double *Tm = pminstance + Lmeasure * i + offsetTm;
        // Received Heat Flux
        for (int j = 0; j < parms.Ly; ++j)
        {
            for (int i = 0; i < parms.Lx; ++i)
            {
                int index = j * parms.Lx + i;
                Tm[index] = T[index];
            }
        }
    }
}
void HFE::EvaluateGPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetInstances().dev();
    double *pminstance = pmeasure->GetInstances().dev();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int Lmeasure = pmeasure->GetMeasureLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    for (int i = 0; i < Lsigma; ++i)
    {
        double *T = psinstance + Lstate * i + offsetT;
        double *Q = psinstance + Lstate * i + offsetQ;
        double *Tm = pminstance + Lmeasure * i + offsetTm;
    }
}

void HFE::SetParms(int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp)
{
    parms.Lx = Lx;
    parms.Ly = Ly;
    parms.Lz = Lz;
    parms.Lt = Lt;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lx;
    parms.dy = Sy / Ly;
    parms.dz = Sz / Lz;
    parms.dt = St / Lt;
    parms.amp = amp;
}

void HFE::SetMemory(Type type)
{
    if (type == Type::CPU)
    {
#if FORWARD_METHOD == 0
        HC::CPU::AllocWorkspaceEuler(workspace, parms);
#elif FORWARD_METHOD == 1
        HC::CPU::AllocWorkspaceRK4(workspace, parms);
#elif FORWARD_METHOD == 2
        HC::CPU::AllocWorkspaceRKF45(workspace, parms);
#endif
    }
    else if (type == Type::GPU)
    {
#if FORWARD_METHOD == 0
        // HC::GPU::AllocWorkspaceEuler(workspace, parms);
#elif FORWARD_METHOD == 1
        // HC::GPU::AllocWorkspaceRK4(workspace, parms);
#elif FORWARD_METHOD == 2
        // HC::GPU::AllocWorkspaceRKF45(workspace, parms);
#endif
    }
}

void HFE::UnsetMemory(Type type)
{
    if (type == Type::CPU)
    {
#if FORWARD_METHOD == 0
        HC::CPU::AllocWorkspaceEuler(workspace, parms);
#elif FORWARD_METHOD == 1
        HC::CPU::AllocWorkspaceRK4(workspace, parms);
#elif FORWARD_METHOD == 2
        HC::CPU::FreeWorkspaceRKF45(workspace);
#endif
    }
    else if (type == Type::GPU)
    {
#if FORWARD_METHOD == 0
        // HC::GPU::AllocWorkspaceEuler(workspace, parms);
#elif FORWARD_METHOD == 1
        // HC::GPU::AllocWorkspaceRK4(workspace, parms);
#elif FORWARD_METHOD == 2
        // HC::GPU::AllocWorkspaceRKF45(workspace, parms);
#endif
    }
}

Data *HFE::GenerateData()
{
    std::string nameT = "Temperature";
    std::string nameQ = "Heat Flux";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lz = parms.Lz;
    dataLoader.Add(nameT, Lx * Ly * Lz);
    dataLoader.Add(nameQ, Lx * Ly);
    double *T = (double *)malloc(3 * sizeof(double) * Lx * Ly * Lz);
    double *cT = T + Lx * Ly * Lz;
    double *nT = cT + Lx * Ly * Lz;
    double *Q = (double *)malloc(3 * sizeof(double) * Lx * Ly);
    double *cQ = Q + Lx * Ly;
    double *nQ = cQ + Lx * Ly;
    for (int i = 0; i < Lx * Ly * Lz; ++i)
    {
        T[i] = 300.0;
        cT[i] = 1.0;
        nT[i] = 1.0;
    }
    for (int i = 0; i < Lx * Ly; ++i)
    {
        Q[i] = 0.0;
        cQ[i] = 1.0;
        nQ[i] = 1.0;
    }
    dataLoader.Link(nameT, T, cT, nT);
    dataLoader.Link(nameQ, Q, cQ, nQ);
    Data *pd = dataLoader.Load();
    free(Q);
    free(T);
    return pd;
}
Measure *HFE::GenerateMeasure()
{
    std::string nameT = "Temperature";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    measureLoader.Add(nameT, Lx * Ly);
    double *nT = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly; ++i)
    {
        nT[i] = 1.0;
    }
    measureLoader.Link(nameT, nT);
    Measure *pm = measureLoader.Load();
    free(nT);
    return pm;
}