#include "../include/hfe.hpp"

Pointer<double> HFE_RM::EvolveMatrixCPU(Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *pwork = workspace.host();

    if (!isValidState)
    {
        HC::RM::CPU::EvolutionJacobianMatrix(pwork + offsetT2, pwork + offsetT2 + (offsetQ - offsetT), pwork + offsetQ2 + (offsetT - offsetQ), pwork + offsetQ2, parms);
        isValidState = true;
    }

    return workspace;
}
Pointer<double> HFE_RM::EvolveMatrixGPU(Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *pwork = workspace.dev();

    if (!isValidState)
    {
        HC::RM::GPU::EvolutionJacobianMatrix(pwork + offsetT2, pwork + offsetT2 + (offsetQ - offsetT), pwork + offsetQ2 + (offsetT - offsetQ), pwork + offsetQ2, parms);
        isValidState = true;
    }

    return workspace;
}
Pointer<double> HFE_RM::EvaluateMatrixCPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *pwork = (workspace + L2).host();

    if (!isValidMeasure)
    {
        HC::RM::CPU::EvaluationMatrix(pwork + offsetT * Lm, pwork + offsetQ * Lm, parms);
        isValidMeasure = true;
    }
    return workspace + L2;
}
Pointer<double> HFE_RM::EvaluateMatrixGPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *pwork = (workspace + L2).dev();

    if (!isValidMeasure)
    {
        HC::RM::GPU::EvaluationMatrix(pwork + offsetT * Lm, pwork + offsetQ * Lm, parms);
        isValidMeasure = true;
    }

    return workspace + L2;
}

Pointer<double> HFE_RM::EvolveStateCPU(Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *ps = pstate->GetStatePointer().host();
    double *pwork = workspace.host();
    double *paux = (double *)malloc(sizeof(double) * L);

    MathCPU::Copy(paux, ps, L);
    if (!isValidState)
    {
        HC::RM::CPU::EvolutionJacobianMatrix(pwork + offsetT2, pwork + offsetT2 + (offsetQ - offsetT), pwork + offsetQ2 + (offsetT - offsetQ), pwork + offsetQ2, parms);
        isValidState = true;
    }

    MathCPU::MatMulNN(0.0, ps, 1.0, pwork, paux, L, L, 1);

    free(paux);
    return pstate->GetStatePointer();
}
Pointer<double> HFE_RM::EvolveStateGPU(Data *pstate)
{
    int L = pstate->GetStateLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *ps = pstate->GetStatePointer().dev();
    double *pwork = workspace.dev();
    double *paux = NULL;
    cudaMallocAsync(&paux, sizeof(double) * L, cudaStreamDefault);

    MathGPU::Copy(paux, ps, L);
    if (!isValidState)
    {
        HC::RM::GPU::EvolutionJacobianMatrix(pwork + offsetT2, pwork + offsetT2 + (offsetQ - offsetT), pwork + offsetQ2 + (offsetT - offsetQ), pwork + offsetQ2, parms);
        isValidState = true;
    }

    MathGPU::MatMulNN(0.0, ps, 1.0, pwork, paux, L, L, 1);

    cudaFreeAsync(paux, cudaStreamDefault);
    return pstate->GetStatePointer();
}
Pointer<double> HFE_RM::EvaluateStateCPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *ps = pstate->GetStatePointer().host();
    double *pm = pmeasure->GetMeasurePointer().host();
    double *pwork = (workspace + L2).host();

    if (!isValidMeasure)
    {
        HC::RM::CPU::EvaluationMatrix(pwork + offsetT * Lm, pwork + offsetQ * Lm, parms);
        isValidMeasure = true;
    }
    MathCPU::MatMulNN(0.0, pm, 1.0, pwork, ps, Lm, L, 1);

    return pmeasure->GetMeasurePointer();
}
Pointer<double> HFE_RM::EvaluateStateGPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *ps = pstate->GetStatePointer().dev();
    double *pm = pmeasure->GetMeasurePointer().dev();
    double *pwork = (workspace + L2).dev();

    if (!isValidMeasure)
    {
        HC::RM::GPU::EvaluationMatrix(pwork + offsetT * Lm, pwork + offsetQ * Lm, parms);
        isValidMeasure = true;
    }
    MathGPU::MatMulNN(0.0, pm, 1.0, pwork, ps, Lm, L, 1);

    return pmeasure->GetMeasurePointer();
}

void HFE_RM::SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp)
{
    parms.Lx = Lx;
    parms.Ly = Ly;
    parms.Lz = 1;
    parms.Lt = Lt;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lx;
    parms.dy = Sy / Ly;
    parms.dz = Sz;
    parms.dt = St / Lt;
    parms.amp = amp;
}
void HFE_RM::SetMemory(Type type)
{
    workspace.alloc(3u * parms.Lx * parms.Ly);
}
void HFE_RM::UnsetMemory(Type type)
{
    workspace.free();
}
Data *HFE_RM::GenerateData()
{
    std::string nameT = "Temperature";
    std::string nameQ = "Heat Flux";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    dataLoader.Add(nameT, Lx * Ly);
    dataLoader.Add(nameQ, Lx * Ly);
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
    dataLoader.Link(nameT, T, cT, nT);
    dataLoader.Link(nameQ, Q, cQ, nQ);
    Data *pd = dataLoader.Load();
    free(Q);
    free(T);
    return pd;
}
Measure *HFE_RM::GenerateMeasure()
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

Pointer<double> HFE::EvolveInstanceCPU(Data *pstate)
{
    double *pinstances = pstate->GetInstances().host();

    int Ls = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int i = 0; i < Lsigma; i++)
    {
#if FORWARD_METHOD == 0
        HC::CPU::Euler(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#elif FORWARD_METHOD == 1
        HC::CPU::RK4(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#elif FORWARD_METHOD == 2
        HC::CPU::RKF45(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#endif
    }
}
Pointer<double> HFE::EvolveInstanceGPU(Data *pstate)
{
    double *pinstances = pstate->GetInstances().dev();

    int Ls = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    for (int i = 0; i < Lsigma; i++)
    {
#if FORWARD_METHOD == 0
        HC::GPU::Euler(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#elif FORWARD_METHOD == 1
        HC::GPU::RK4(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#elif FORWARD_METHOD == 2
        HC::GPU::RKF45(pinstances + offsetT + Ls * i, pinstances + offsetQ + Ls * i, workspace, parms);
#endif
    }
}
Pointer<double> HFE::EvaluateInstanceCPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetInstances().host();
    double *pminstance = pmeasure->GetInstances().host();

    int Ls = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    for (int i = 0; i < Lsigma; i++)
    {
        MathCPU::Copy(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i, parms.Lx * parms.Ly);
    }
}
Pointer<double> HFE::EvaluateInstanceGPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetInstances().dev();
    double *pminstance = pmeasure->GetInstances().dev();

    int Ls = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    for (int i = 0; i < Lsigma; i++)
    {
        MathGPU::Copy(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i, parms.Lx * parms.Ly);
    }
}

Pointer<double> HFE::EvolveStateCPU(Data *pstate)
{
    double *ps = pstate->GetStatePointer().host();

    int Ls = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
#if FORWARD_METHOD == 0
    HC::CPU::Euler(ps + offsetT, ps + offsetQ, workspace, parms);
#elif FORWARD_METHOD == 1
    HC::CPU::RK4(ps + offsetT, ps + offsetQ, workspace, parms);
#elif FORWARD_METHOD == 2
    HC::CPU::RKF45(ps + offsetT, ps + offsetQ, workspace, parms);
#endif
}
Pointer<double> HFE::EvolveStateGPU(Data *pstate)
{
    double *ps = pstate->GetStatePointer().dev();

    int Ls = pstate->GetStateLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
#if FORWARD_METHOD == 0
    HC::GPU::Euler(ps + offsetT, ps + offsetQ, workspace, parms);
#elif FORWARD_METHOD == 1
    HC::GPU::RK4(ps + offsetT, ps + offsetQ, workspace, parms);
#elif FORWARD_METHOD == 2
    HC::GPU::RKF45(ps + offsetT, ps + offsetQ, workspace, parms);
#endif
}
Pointer<double> HFE::EvaluateStateCPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetStatePointer().host();
    double *pminstance = pmeasure->GetMeasurePointer().host();

    int Ls = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    MathCPU::Copy(pminstance + offsetTm, psinstance + offsetT, parms.Lx * parms.Ly);
}
Pointer<double> HFE::EvaluateStateGPU(Measure *pmeasure, Data *pstate)
{
    double *psinstance = pstate->GetStatePointer().dev();
    double *pminstance = pmeasure->GetMeasurePointer().dev();

    int Ls = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int Lsigma = pstate->GetSigmaLength();
    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetTm = pmeasure->GetOffset("Temperature");
    MathCPU::Copy(pminstance + offsetTm, psinstance + offsetT, parms.Lx * parms.Ly);
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