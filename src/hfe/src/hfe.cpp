#include "../include/hfe.hpp"

void HFE::EvolveCPU(Data *pstate)
{
    double *pinstance = pstate->GetInstances().host();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int i = 0; i < Lsigma; ++i)
    {
        double *T = pinstance + Lx * i + offsetT;
        double *Q = pinstance + Lx * i + offsetQ;
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
        double *T = pinstance + Lx * i + offsetT;
        double *Q = pinstance + Lx * i + offsetQ;
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
        double *T = psinstance + Lx * i + offsetT;
        double *Q = psinstance + Lx * i + offsetQ;
        double *Tm = pminstance + Lx * i + offsetQ;
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
        double *T = psinstance + Lx * i + offsetT;
        double *Q = psinstance + Lx * i + offsetQ;
        double *Tm = pminstance + Lx * i + offsetTm;
    }
}

Data &HFE::GenerateData()
{
    std::string nameT = "Temperature";
    std::string nameQ = "Heat Flux";
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
    Data d = dataLoader.Load();
    free(Q);
    free(T);
    return d;
}
Measure &HFE::GenerateMeasure()
{
    std::string nameT = "Temperature";
    measureLoader.Add(nameT, Lx * Ly);
    double *nT = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly * Lz; ++i)
    {
        nT[i] = 1.0;
    }
    measureLoader.Link(nameT, nT);
    Measure m = measureLoader.Load();
    return m;
}