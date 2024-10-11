#include "../include/hfe2D.hpp"

Pointer<double> HFE2D::EvolveMatrixCPU(Data *pstate, Control *pcontrol)
{
    int L = pstate->GetStateLength();
    int Lu = pcontrol->GetControlLength();
    int L2 = L * L;
    int L2u = L * Lu;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *pwork = workspaceState.host();
    double *puwork = pwork + L2;

    if (!isValidState)
    {
        HC2D::CPU::EvolutionMatrix(parms, pwork, puwork, offsetQ - offsetT);
        isValidState = true;
    }

    return workspaceState;
}
Pointer<double> HFE2D::EvolveMatrixGPU(Data *pstate, Control *pcontrol)
{
    int L = pstate->GetStateLength();
    int Lu = pcontrol->GetControlLength();
    int L2 = L * L;
    int L2u = L * Lu;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *pwork = workspaceState.dev();
    double *puwork = pwork + L2;

    if (!isValidState)
    {
        HC2D::GPU::EvolutionMatrix(parms, pwork, puwork, offsetQ - offsetT);
        isValidState = true;
    }

    return workspaceState;
}
Pointer<double> HFE2D::EvaluateMatrixCPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *pwork = workspaceMeasure.host();

    if (!isValidMeasure)
    {
        HC2D::CPU::EvaluationMatrix(parms, pwork + offsetT * Lm, pwork + offsetQ * Lm);
        isValidMeasure = true;
    }
    return workspaceMeasure;
}
Pointer<double> HFE2D::EvaluateMatrixGPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *pwork = workspaceMeasure.dev();

    if (!isValidMeasure)
    {
        HC2D::GPU::EvaluationMatrix(parms, pwork + offsetT * Lm, pwork + offsetQ * Lm);
        isValidMeasure = true;
    }

    return workspaceMeasure;
}

Pointer<double> HFE2D::EvolveStateCPU(Data *pstate, Control *pcontrol)
{
    int L = pstate->GetStateLength();
    int Lu = pcontrol->GetControlLength();
    int L2 = L * L;
    int L2u = L * Lu;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *ps = pstate->GetStatePointer().host();
    double *pc = pcontrol->GetControlPointer().host();
    double *pwork = workspaceState.host();
    double *puwork = pwork + L2;
    double *paux = (double *)malloc(sizeof(double) * L);

    MathCPU::Copy(paux, ps, L);
    if (!isValidState)
    {
        HC2D::CPU::EvolutionMatrix(parms, pwork, puwork, offsetQ - offsetT);;
        isValidState = true;
    }

    MathCPU::MatMulNN(0.0, ps, 1.0, pwork, paux, L, L, 1);
    MathCPU::MatMulNN(1.0, ps, 1.0, puwork, pc, L, Lu, 1);

    free(paux);
    return pstate->GetStatePointer();
}
Pointer<double> HFE2D::EvolveStateGPU(Data *pstate, Control *pcontrol)
{
    int L = pstate->GetStateLength();
    int Lu = pcontrol->GetControlLength();
    int L2 = L * L;
    int L2u = L * Lu;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");
    int offsetT2 = pstate->GetOffset2("Temperature");
    int offsetQ2 = pstate->GetOffset2("Heat Flux");

    double *ps = pstate->GetStatePointer().dev();
    double *pc = pcontrol->GetControlPointer().dev();
    double *pwork = workspaceState.dev();
    double *puwork = pwork + L2u;
    double *paux = NULL;
    cudaMallocAsync(&paux, sizeof(double) * L, cudaStreamDefault);

    MathGPU::Copy(paux, ps, L);
    if (!isValidState)
    {
        HC2D::GPU::EvolutionMatrix(parms, pwork, puwork, offsetQ - offsetT);
        isValidState = true;
    }

    MathGPU::MatMulNN(0.0, ps, 1.0, pwork, paux, L, L, 1);
    MathGPU::MatMulNN(1.0, ps, 1.0, puwork, pc, L, Lu, 1);

    cudaFreeAsync(paux, cudaStreamDefault);
    return pstate->GetStatePointer();
}
Pointer<double> HFE2D::EvaluateStateCPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *ps = pstate->GetStatePointer().host();
    double *pm = pmeasure->GetMeasurePointer().host();
    double *pwork = workspaceMeasure.host();

    if (!isValidMeasure)
    {
        HC2D::CPU::EvaluationMatrix(parms, pwork + offsetT * Lm, pwork + offsetQ * Lm);
        isValidMeasure = true;
    }
    MathCPU::MatMulNN(0.0, pm, 1.0, pwork, ps, Lm, L, 1);

    return pmeasure->GetMeasurePointer();
}
Pointer<double> HFE2D::EvaluateStateGPU(Measure *pmeasure, Data *pstate)
{
    int L = pstate->GetStateLength();
    int Lm = pmeasure->GetMeasureLength();
    int L2 = L * L;

    int offsetT = pstate->GetOffset("Temperature");
    int offsetQ = pstate->GetOffset("Heat Flux");

    double *ps = pstate->GetStatePointer().dev();
    double *pm = pmeasure->GetMeasurePointer().dev();
    double *pwork = workspaceMeasure.dev();

    if (!isValidMeasure)
    {
        HC2D::GPU::EvaluationMatrix(parms, pwork + offsetT * Lm, pwork + offsetQ * Lm);
        isValidMeasure = true;
    }
    MathGPU::MatMulNN(0.0, pm, 1.0, pwork, ps, Lm, L, 1);

    return pmeasure->GetMeasurePointer();
}

void HFE2D::SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double h)
{
    parms.Lx = Lx;
    parms.Ly = Ly;
    parms.Lt = Lt;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lx;
    parms.dy = Sy / Ly;
    parms.dt = St / Lt;
    parms.amp = amp;
    parms.h = h;
}
void HFE2D::SetMemory(Type type)
{
    int Lxy = parms.Lx * parms.Ly;
    workspaceState.alloc(4u * Lxy * Lxy + 2u * Lxy);
    workspaceMeasure.alloc(2u * Lxy * Lxy);
    isValidState = false;
    isValidMeasure = false;
}
void HFE2D::UnsetMemory(Type type)
{
    workspaceState.free();
    workspaceMeasure.free();
}
Data *HFE2D::GenerateData()
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
Control *HFE2D::GenerateControl()
{
    std::string nameU = "Ambient Temperature";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    controlLoader.Add(nameU, 1);
    double *U = (double *)malloc(sizeof(double));
    U[0] = 300.0;
    controlLoader.Link(nameU, U);
    Control *pc = controlLoader.Load();
    free(U);
    return pc;
}
Measure *HFE2D::GenerateMeasure()
{
    std::string nameT = "Temperature";
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    measureLoader.Add(nameT, Lx * Ly);
    double *nT = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly; ++i)
    {
        nT[i] = 25.0;
    }
    measureLoader.Link(nameT, nT);
    Measure *pm = measureLoader.Load();
    free(nT);
    return pm;
}