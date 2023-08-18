#include "../include/hfe.hpp"

inline double C(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

inline double K(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

double DiffK(double TN_in, double T_in, double TP_in, double delta_in)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (auxN + auxP) / delta_in;
}

void HFE::EvolveCPU(Data *pstate)
{
    double *pinstance = pstate->GetInstances().host();
    int Lstate = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int s = 0; s < Lsigma; ++s)
    {
        double *T = pinstance + Lstate * s + offsetT;
        double *Q = pinstance + Lstate * s + offsetQ;
        double *aux = (double *)malloc(sizeof(double) * Lx * Ly * (Lz + 1));
        // Diffusion Contribution
        for (int k = 0; k < Lz; ++k)
        {
            for (int j = 0; j < Ly; ++j)
            {
                for (int i = 0; i < Lx; ++i)
                {
                    int index = (k * Ly + j) * Lx + i;
                    double T0 = T[index];
                    double TiP = (i < Lx - 1) ? T[index + 1] : T0;
                    double TiN = (i > 0) ? T[index - 1] : T0;
                    double TjP = (j < Ly - 1) ? T[index + Lx] : T0;
                    double TjN = (j > 0) ? T[index - Lx] : T0;
                    double TkP = (k < Lz - 1) ? T[index + Ly * Lx] : T0;
                    double TkN = (k > 0) ? T[index - Ly * Lx] : T0;
                    aux[index] = (DiffK(TiN, T0, TiP, dx) + DiffK(TjN, T0, TjP, dy) + DiffK(TkN, T0, TkP, dz));
                }
            }
        }
        // Received Heat Flux
        int offset = (Lz - 1) * Ly * Lz;
        for (int j = 0; j < Ly; ++j)
        {
            for (int i = 0; i < Lx; ++i)
            {
                int index = j * Lx + i;
                aux[offset + index] += (amp / dz) * Q[index];
            }
        }
        // Calculate Temporal Derivative
        for (int k = 0; k < Lz; ++k)
        {
            for (int j = 0; j < Ly; ++j)
            {
                for (int i = 0; i < Lx; ++i)
                {
                    int index = (k * Ly + j) * Lx + i;
                    aux[index] /= C(T[index]);
                }
            }
        }
        // Random Step of Heat Flux
        int offset = Lz * Ly * Lz;
        for (int j = 0; j < Ly; ++j)
        {
            for (int i = 0; i < Lx; ++i)
            {
                int index = j * Lx + i;
                aux[offset + index] = distribution(generator);
            }
        }
        free(aux);
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
        double *Tm = pminstance + Lstate * i + offsetQ;
        // Received Heat Flux
        for (int j = 0; j < Ly; ++j)
        {
            for (int i = 0; i < Lx; ++i)
            {
                int index = j * Lx + i;
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