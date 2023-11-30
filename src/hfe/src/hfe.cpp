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
        MathCPU::Mul(pwork, parms.dt, L2);
        MathCPU::AddIdentity(pwork, L, L);
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
        MathGPU::Mul(pwork, parms.dt, L2);
        MathGPU::AddIdentity(pwork, L, L);
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
        MathCPU::Mul(pwork, parms.dt, L2);
        MathCPU::AddIdentity(pwork, L, L);
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
        MathGPU::Mul(pwork, parms.dt, L2);
        MathGPU::AddIdentity(pwork, L, L);
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

void HFE_RM::SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double T_ref)
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
    parms.T_ref = T_ref;
}
void HFE_RM::SetMemory(Type type)
{
    int Lxy = parms.Lx * parms.Ly;
    workspace.alloc(6u * Lxy * Lxy);
    isValidState = false;
    isValidMeasure = false;
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
    return pstate->GetInstances();
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
    return pstate->GetInstances();
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
        MathCPU::Mul(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i, 3.0 / 2.0, parms.Lx * parms.Ly);
        MathCPU::LRPO(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i + parms.Lx * parms.Ly, -1.0 / 2.0, parms.Lx * parms.Ly);
    }
    return pmeasure->GetInstances();
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
        MathGPU::Mul(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i, 3.0 / 2.0, parms.Lx * parms.Ly);
        MathGPU::LRPO(pminstance + offsetTm + Lm * i, psinstance + offsetT + Ls * i + parms.Lx * parms.Ly, -1.0 / 2.0, parms.Lx * parms.Ly);
    }
    return pmeasure->GetInstances();
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
    return pstate->GetStatePointer();
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
    return pstate->GetStatePointer();
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
    MathCPU::Mul(pminstance + offsetTm, psinstance + offsetT, 3.0 / 2.0, parms.Lx * parms.Ly);
    MathCPU::LRPO(pminstance + offsetTm, psinstance + offsetT + parms.Lx * parms.Ly, -1.0 / 2.0, parms.Lx * parms.Ly);

    return pmeasure->GetMeasurePointer();
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
    MathGPU::Mul(pminstance + offsetTm, psinstance + offsetT, 3.0 / 2.0, parms.Lx * parms.Ly);
    MathGPU::LRPO(pminstance + offsetTm, psinstance + offsetT + parms.Lx * parms.Ly, -1.0 / 2.0, parms.Lx * parms.Ly);
    return pmeasure->GetMeasurePointer();
}

void HFE::SetParms(int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp, double T_ref)
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
    parms.T_ref = T_ref;
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

void HFE_AEM::SetModel(HFE_RM *rm, HFE *cm)
{
    AEM::SetModel(rm, cm);
}

void HFE_AEM::SetMemory(Type type)
{

    int Lrx = ((HFE_RM *)reducedModel)->parms.Lx;
    int Lry = ((HFE_RM *)reducedModel)->parms.Ly;
    int Lrz = ((HFE_RM *)reducedModel)->parms.Lz;
    int Lcx = ((HFE *)completeModel)->parms.Lx;
    int Lcy = ((HFE *)completeModel)->parms.Ly;
    int Lcz = ((HFE *)completeModel)->parms.Lz;

    int Lcs = Lcx * Lcy * (Lcz + 1);
    int Lcm = Lcx * Lcy;
    int Lrs = Lrx * Lry * (Lrz + 1);
    int Lrm = Lrx * Lry;

    workspace.alloc(2 * Lcx * Lcy);
    AEM::SetMemory(Lcs, Lcm, Lrs, Lrm, type);
    this->type = type;
}
void HFE_AEM::UnsetMemory(Type type)
{
    AEM::UnsetMemory(type);
    workspace.free();
}

void HFE_AEM::R2C(Data *prState, Data *pcState)
{
    Pointer<double> rx = prState->GetStatePointer();
    Pointer<double> cx = pcState->GetStatePointer();

    int rOffsetT = prState->GetOffset("Temperature");
    int rOffsetQ = prState->GetOffset("Heat Flux");
    int cOffsetT = pcState->GetOffset("Temperature");
    int cOffsetQ = pcState->GetOffset("Heat Flux");

    double Sx = ((HFE_RM *)reducedModel)->parms.Sx;
    double Sy = ((HFE_RM *)reducedModel)->parms.Sy;
    double Sz = ((HFE_RM *)reducedModel)->parms.Sz;

    int Lrx = ((HFE_RM *)reducedModel)->parms.Lx;
    int Lry = ((HFE_RM *)reducedModel)->parms.Ly;
    int Lrz = ((HFE_RM *)reducedModel)->parms.Lz;
    int Lcx = ((HFE *)completeModel)->parms.Lx;
    int Lcy = ((HFE *)completeModel)->parms.Ly;
    int Lcz = ((HFE *)completeModel)->parms.Lz;

    Interpolation::Rescale(rx + rOffsetT, Lrx, Lry, workspace, Lcx, Lcy, Sx, Sy, type);
    Interpolation::Rescale(rx + rOffsetQ, Lrx, Lry, cx + cOffsetQ, Lcx, Lcy, Sx, Sy, type);

    double KT = HC::K(((HFE_RM *)reducedModel)->parms.T_ref);
    double amp = ((HFE_RM *)reducedModel)->parms.amp;
    Pointer<double> pwork = workspace + Lcx * Lcy;
    if (type == Type::GPU)
    {
        workspace.copyDev2Host(Lcx * Lcy);
        (cx + cOffsetQ).copyDev2Host(Lcy * Lcy);
        cudaDeviceSynchronize();
    }
    double *pw0_h = workspace.host();
    double *pw1_h = (cx + cOffsetQ).host();
    double *pw2_h = pwork.host();
    for (int i = 0; i < Lcx * Lcy; i++)
    {
        double val = pw0_h[i] + amp * Sz * pw1_h[i] / (3 * KT);
        pw2_h[i] = HC::K(val);
    }
    if (type == Type::GPU)
    {
        pwork.copyHost2Dev(Lcx * Lcy);
    }
    Math::Div(pwork, cx + cOffsetQ, pwork, Lcx * Lcy, type);

    for (int k = 0; k < Lcz; k++)
    {
        double z = (k + 0.5) * Sz / Lcz;
        double temp = (-Sz / 6.0 + z * z / (2.0 * Sz));
        Math::Copy(cx + cOffsetT + k * Lcx * Lcy, workspace, Lcx * Lcy, type);
        Math::LRPO(cx + cOffsetT + k * Lcx * Lcy, pwork, amp * temp, Lcx * Lcy, type);
    }
}
void HFE_AEM::R2C(Measure *prMeasure, Measure *pcMeasure)
{
    Pointer<double> ry = prMeasure->GetMeasurePointer();
    Pointer<double> cy = pcMeasure->GetMeasurePointer();

    double Sx = ((HFE_RM *)reducedModel)->parms.Sx;
    double Sy = ((HFE_RM *)reducedModel)->parms.Sy;
    double Sz = ((HFE_RM *)reducedModel)->parms.Sz;

    int Lrx = ((HFE_RM *)reducedModel)->parms.Lx;
    int Lry = ((HFE_RM *)reducedModel)->parms.Ly;
    int Lrz = ((HFE_RM *)reducedModel)->parms.Lz;
    int Lcx = ((HFE *)completeModel)->parms.Lx;
    int Lcy = ((HFE *)completeModel)->parms.Ly;
    int Lcz = ((HFE *)completeModel)->parms.Lz;

    Interpolation::Rescale(ry, Lrx, Lry, cy, Lcx, Lcy, Sx, Sy, type);
}
void HFE_AEM::C2R(Data *pcState, Data *prState)
{
    Pointer<double> rx = prState->GetStatePointer();
    Pointer<double> cx = pcState->GetStatePointer();

    int rOffsetT = prState->GetOffset("Temperature");
    int rOffsetQ = prState->GetOffset("Heat Flux");
    int cOffsetT = pcState->GetOffset("Temperature");
    int cOffsetQ = pcState->GetOffset("Heat Flux");

    double Sx = ((HFE_RM *)reducedModel)->parms.Sx;
    double Sy = ((HFE_RM *)reducedModel)->parms.Sy;
    double Sz = ((HFE_RM *)reducedModel)->parms.Sz;

    int Lrx = ((HFE_RM *)reducedModel)->parms.Lx;
    int Lry = ((HFE_RM *)reducedModel)->parms.Ly;
    int Lrz = ((HFE_RM *)reducedModel)->parms.Lz;
    int Lcx = ((HFE *)completeModel)->parms.Lx;
    int Lcy = ((HFE *)completeModel)->parms.Ly;
    int Lcz = ((HFE *)completeModel)->parms.Lz;

    Math::Mean(workspace, cx + cOffsetT, Lcx * Lcy, Lcz, type);
    Interpolation::Rescale(workspace, Lcx, Lcy, rx + rOffsetT, Lrx, Lry, Sx, Sy, type);
    Interpolation::Rescale(cx + cOffsetQ, Lcx, Lcy, rx + rOffsetQ, Lrx, Lry, Sx, Sy, type);
}
void HFE_AEM::C2R(Measure *pcMeasure, Measure *prMeasure)
{
    Pointer<double> ry = prMeasure->GetMeasurePointer();
    Pointer<double> cy = pcMeasure->GetMeasurePointer();

    double Sx = ((HFE_RM *)reducedModel)->parms.Sx;
    double Sy = ((HFE_RM *)reducedModel)->parms.Sy;
    double Sz = ((HFE_RM *)reducedModel)->parms.Sz;

    int Lrx = ((HFE_RM *)reducedModel)->parms.Lx;
    int Lry = ((HFE_RM *)reducedModel)->parms.Ly;
    int Lrz = ((HFE_RM *)reducedModel)->parms.Lz;
    int Lcx = ((HFE *)completeModel)->parms.Lx;
    int Lcy = ((HFE *)completeModel)->parms.Ly;
    int Lcz = ((HFE *)completeModel)->parms.Lz;

    Interpolation::Rescale(cy, Lcx, Lcy, ry, Lrx, Lry, Sx, Sy, type);
}
int HFE_AEM::GetSampleLength(int Lrs)
{
    return 10;
}
void HFE_AEM::SampleStates(Type type)
{
    unsigned Lrs = prState->GetStateLength();
    unsigned Lrm = prMeasure->GetMeasureLength();
    unsigned Lcs = pcState->GetStateLength();
    unsigned Lcm = pcMeasure->GetMeasureLength();
    unsigned Lrs2 = Lrs * Lrs;
    unsigned Ls = (2 * Lrs + Lcs);
    unsigned Lm = (2 * Lrm + Lcm);
    unsigned N = GetSampleLength(0);
    Pointer<double> workspace;
    workspace.alloc(Lrs2 + Lrs * N);
    Math::Zero(workspace, Lrs2, type);
    Math::Zero(samplesState, Ls * N, type);
    Math::Zero(samplesMeasure, Lm * N, type);

    Math::CholeskyDecomposition(workspace, prState->GetStateCovariancePointer(), Lrs, type);
    Math::Iterate(Math::Copy, samplesState, prState->GetStatePointer(), Lrs, N, Lrs, 0, 0, 0, type);
    Random::SampleNormal(workspace + Lrs2, Lrs * N, 0.0, 1.0, type);
    Math::MatMulNN(1.0, samplesState, 1.0, workspace, workspace + Lrs2, Lrs, Lrs, N, type);
    workspace.free();
}