#include "../include/hfe.hpp"

void HFE::EvolveCPU(Data *pstate, int index)
{
    double *pinstance = (index < 0) ? pstate->GetStatePointer().host() : pstate->GetInstances().host();
    index = (index < 0) ? 0 : index;
    int Lstate = pstate->GetStateLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
#if FORWARD_METHOD == 0
    HC::CPU::Euler(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#elif FORWARD_METHOD == 1
    HC::CPU::RK4(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#elif FORWARD_METHOD == 2
    HC::CPU::RKF45(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#endif
}
void HFE::EvolveGPU(Data *pstate, int index)
{
    double *pinstance = (index < 0) ? pstate->GetStatePointer().dev() : pstate->GetInstances().dev();
    index = (index < 0) ? 0 : index;
    int Lstate = pstate->GetStateLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
#if FORWARD_METHOD == 0
    HC::GPU::Euler(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#elif FORWARD_METHOD == 1
    HC::GPU::RK4(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#elif FORWARD_METHOD == 2
    HC::GPU::RKF45(pinstance + offsetT + Lstate * index, pinstance + offsetQ + Lstate * index, workspace, parms);
#endif
}
void HFE::EvaluateCPU(Measure *pmeasure, Data *pstate, int index)
{
    double *psinstance = (index < 0) ? pstate->GetStatePointer().host() : pstate->GetInstances().host();
    double *pminstance = (index < 0) ? pmeasure->GetMeasurePointer().host() : pmeasure->GetInstances().host();
    index = (index < 0) ? 0 : index;
    int Lstate = pstate->GetStateLength();
    int Lmeasure = pmeasure->GetMeasureLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    MathCPU::Copy(pminstance + Lmeasure * index + offsetTm, psinstance + Lstate * index + offsetT, parms.Lx * parms.Ly);
}
void HFE::EvaluateGPU(Measure *pmeasure, Data *pstate, int index)
{
    double *psinstance = (index < 0) ? pstate->GetStatePointer().dev() : pstate->GetInstances().dev();
    double *pminstance = (index < 0) ? pmeasure->GetMeasurePointer().dev() : pmeasure->GetInstances().dev();
    index = (index < 0) ? 0 : index;
    int Lstate = pstate->GetStateLength();
    int Lmeasure = pmeasure->GetMeasureLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    MathGPU::Copy(pminstance + Lmeasure * index + offsetTm, psinstance + Lstate * index + offsetT, parms.Lx * parms.Ly);
}

void HFE::EvolutionCPU(Pointer<double> m_o, Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;
    double *pm = m_o.host();
    double *pm_I = (double *)malloc(L2 * sizeof(double));
    double *pm_P = (double *)malloc(L2 * sizeof(double));
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");
    MathCPU::Identity(pm, L, L);
    MathCPU::Identity(pm_I, L, L);
    MathCPU::Zero(pm_P, L * L);
    HC::RM::CPU::EvolutionJacobianMatrix(pm_P + offsetT2, pm_P + offsetT2 + (offsetQ - offsetT), pm_P + offsetQ2 + (offsetT - offsetQ), pm_P + offsetQ2, parms);

#if FORWARD_METHOD == 0
    MathCPU::LRPO(pm, pm_P, parms.dt, L2);
#elif FORWARD_METHOD == 1
    MathCPU::LRPO(pm, pm_P, parms.dt / 4.0, L2);
    for (int i = 1; i < 4; i++)
    {
        MathCPU::MatMulNN(0.0, pm_I, 1.0, pm_P, pm, L, L, L);
        MathCPU::Identity(pm, L, L);
        MathCPU::LRPO(pm, pm_I, parms.dt / (4.0 - i), L2);
    }
#elif FORWARD_METHOD == 2
    MathCPU::LRPO(pm, pm_P, parms.dt / 4.0, L2);
    for (int i = 1; i < 4; i++)
    {
        MathCPU::MatMulNN(0.0, pm_I, 1.0, pm_P, pm, L, L, L);
        MathCPU::Identity(pm, L, L);
        MathCPU::LRPO(pm, pm_I, parms.dt / (4.0 - i), L2);
    }
#endif
}
void HFE::EvolutionGPU(Pointer<double> m_o, Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;
    double *pm = m_o.dev();
    double *pm_I = NULL;
    double *pm_P = NULL;
    cudaMallocAsync(&pm_I, L2 * sizeof(double), cudaStreamDefault);
    cudaMallocAsync(&pm_P, L2 * sizeof(double), cudaStreamDefault);
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");
    MathGPU::Identity(pm, L, L);
    MathGPU::Identity(pm_I, L, L);
    MathGPU::Zero(pm_P, L * L);
    HC::RM::GPU::EvolutionJacobianMatrix(pm_P + offsetT2, pm_P + offsetT2 + (offsetQ - offsetT), pm_P + offsetQ2 + (offsetT - offsetQ), pm_P + offsetQ2, parms);

#if FORWARD_METHOD == 0
    MathGPU::LRPO(pm, pm_P, parms.dt, L2);
#elif FORWARD_METHOD == 1
    MathGPU::LRPO(pm, pm_P, parms.dt / 4.0, L2);
    for (int i = 1; i < 4; i++)
    {
        MathGPU::MatMulNN(0.0, pm_I, 1.0, pm_P, pm, L, L, L);
        MathGPU::Identity(pm, L, L);
        MathGPU::LRPO(pm, pm_I, parms.dt / (4.0 - i), L2);
    }
#elif FORWARD_METHOD == 2
    MathGPU::LRPO(pm, pm_P, parms.dt / 4.0, L2);
    for (int i = 1; i < 4; i++)
    {
        MathGPU::MatMulNN(0.0, pm_I, 1.0, pm_P, pm, L, L, L);
        MathGPU::Identity(pm, L, L);
        MathGPU::LRPO(pm, pm_I, parms.dt / (4.0 - i), L2);
    }
#endif
}
void HFE::EvaluationCPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate)
{
    double *pm = m_o.host();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");
    int Lm = pmeasure->GetMeasureLength();
    double *mTT = pm + offsetT * Lm;
    double *mQT = pm + offsetQ * Lm;

    HC::RM::CPU::EvaluationMatrix(mTT, mQT, parms);
}
void HFE::EvaluationGPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate)
{
    double *pm = m_o.dev();
    double *pm_I = NULL;
    double *pm_P = NULL;
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");
    int Lm = pmeasure->GetMeasureLength();
    double *mTT = pm + offsetT * Lm;
    double *mQT = pm + offsetQ * Lm;
    HC::RM::GPU::EvaluationMatrix(mTT, mQT, parms);
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
        HC::GPU::AllocWorkspaceEuler(workspace, parms);
#elif FORWARD_METHOD == 1
        HC::GPU::AllocWorkspaceRK4(workspace, parms);
#elif FORWARD_METHOD == 2
        HC::GPU::AllocWorkspaceRKF45(workspace, parms);
#endif
    }
}

void HFE::UnsetMemory(Type type)
{
    if (type == Type::CPU)
    {
#if FORWARD_METHOD == 0
        HC::CPU::FreeWorkspaceEuler(workspace);
#elif FORWARD_METHOD == 1
        HC::CPU::FreeWorkspaceRK4(workspace);
#elif FORWARD_METHOD == 2
        HC::CPU::FreeWorkspaceRKF45(workspace);
#endif
    }
    else if (type == Type::GPU)
    {
#if FORWARD_METHOD == 0
        HC::GPU::FreeWorkspaceEuler(workspace);
#elif FORWARD_METHOD == 1
        HC::GPU::FreeWorkspaceRK4(workspace);
#elif FORWARD_METHOD == 2
        HC::GPU::FreeWorkspaceRKF45(workspace);
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
    Model::dataLoader.Add(nameT, Lx * Ly * Lz);
    Model::dataLoader.Add(nameQ, Lx * Ly);
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
    Model::dataLoader.Link(nameT, T, cT, nT);
    Model::dataLoader.Link(nameQ, Q, cQ, nQ);
    Data *pd = Model::dataLoader.Load();
    free(Q);
    free(T);
    return pd;
}
Measure *HFE::GenerateMeasure()
{
    std::string nameT = "Temperature";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    Model::measureLoader.Add(nameT, Lx * Ly);
    double *nT = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly; ++i)
    {
        nT[i] = 1.0;
    }
    Model::measureLoader.Link(nameT, nT);
    Measure *pm = Model::measureLoader.Load();
    free(nT);
    return pm;
}

Data *HFE::GenerateLinearData()
{
    std::string nameT = "Temperature";
    std::string nameQ = "Heat Flux";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    LinearModel::dataLoader.Add(nameT, Lx * Ly);
    LinearModel::dataLoader.Add(nameQ, Lx * Ly);
    double *T = (double *)malloc(3 * sizeof(double) * Lx * Ly);
    double *cT = T + Lx * Ly;
    double *nT = cT + Lx * Ly;
    double *Q = (double *)malloc(3 * sizeof(double) * Lx * Ly);
    double *cQ = Q + Lx * Ly;
    double *nQ = cQ + Lx * Ly;
    for (int i = 0; i < Lx * Ly; ++i)
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
    LinearModel::dataLoader.Link(nameT, T, cT, nT);
    LinearModel::dataLoader.Link(nameQ, Q, cQ, nQ);
    Data *pd = LinearModel::dataLoader.Load();
    free(Q);
    free(T);
    return pd;
}
Measure *HFE::GenerateLinearMeasure()
{
    std::string nameT = "Temperature";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    LinearModel::measureLoader.Add(nameT, Lx * Ly);
    double *nT = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly; ++i)
    {
        nT[i] = 1.0;
    }
    LinearModel::measureLoader.Link(nameT, nT);
    Measure *pm = LinearModel::measureLoader.Load();
    free(nT);
    return pm;
}