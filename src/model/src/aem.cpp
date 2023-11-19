#include "../include/aem.hpp"

#include <fstream>

int it = 0;

void AEM::SetMemory(int Lcs, int Lcm, int Lrs, int Lrm, Type type)
{
    errorState.alloc(Lrs);
    errorMeasure.alloc(Lrm);
    covarState.alloc(Lrs * Lrs);
    covarMeasure.alloc(Lrm * Lrm);
    int Ns = GetSampleLength(Lrs);
    samplesState.alloc(Ns * (2 * Lrs + Lcs));
    samplesMeasure.alloc(Ns * (2 * Lrm + Lcm));
}
void AEM::UnsetMemory(Type type)
{
    samplesMeasure.free();
    samplesState.free();
    covarMeasure.free();
    covarState.free();
    errorMeasure.free();
    errorState.free();
}

int AEM::GetSampleLength(int Lrs)
{
    return 2 * Lrs + 1;
}

void AEM::SampleStates(Type type)
{
    unsigned Lrs = prState->GetStateLength();
    unsigned Lrm = prMeasure->GetMeasureLength();
    unsigned Lcs = pcState->GetStateLength();
    unsigned Lcm = pcMeasure->GetMeasureLength();
    unsigned Lrs2 = Lrs * Lrs;
    unsigned Ls = (2 * Lrs + Lcs);
    unsigned Lm = (2 * Lrm + Lcm);
    unsigned N = GetSampleLength(Lrs);
    unsigned NLs = Ls * N;
    unsigned NLm = Lm * N;
    Math::Zero(samplesState, NLs, type);
    Math::Zero(samplesMeasure, NLm, type);

    Pointer<double> workspace;
    workspace.alloc(Lrs2);
    Math::Zero(workspace, Lrs2, type);
    Math::CholeskyDecomposition(workspace, prState->GetStateCovariancePointer(), Lrs, type);
    Math::Iterate(Math::Copy, samplesState, prState->GetStatePointer(), Lrs, N, Lrs, 0, 0, 0, type);
    Math::Iterate(Math::Add, samplesState, workspace, Lrs, Lrs, Lrs, Lrs, Lrs, 0, type);
    Math::Iterate(Math::Sub, samplesState, workspace, Lrs, Lrs, Lrs, Lrs, Lrs * (Lrs + 1), 0, type);
    workspace.free();
}

void AEM::SetModel(Model *rm, Model *cm)
{
    reducedModel = rm;
    completeModel = cm;
}

void AEM::CorrectEstimation(Data *pstate, Type type)
{
    int L = pstate->GetStateLength();
    Math::Add(pstate->GetStatePointer(), errorState, L, type);
    //Math::Add(pstate->GetStateCovariancePointer(), covarState, L * L, type);
}

void AEM::CorrectEvaluation(Measure *pmeasure, Data *pstate, Type type)
{
    int L = pmeasure->GetMeasureLength();
    Math::Add(pmeasure->GetMeasurePointer(), errorMeasure, L, type);
    Math::Add(pmeasure->GetMeasureCovariancePointer(), covarMeasure, L * L, type);
}

void PrintMatrix(std::string name, Pointer<double> mat, int lengthI, int lengthJ, Type type)
{
    if (type == Type::GPU)
    {
        cudaDeviceSynchronize();
        mat.copyDev2Host(lengthI * lengthJ);
    }
    double *p = mat.host();

    std::ofstream fp;
    fp.open(name + std::to_string(it) + ".csv");
    for (int i = 0; i < lengthI; ++i)
    {
        for (int j = 0; j < lengthJ; ++j)
        {
            fp << p[j * lengthI + i];
            if (j < lengthJ)
            {
                fp << ",";
            }
        }
        fp << "\n";
    }
    fp.close();
}

Pointer<double> AEM::Evolve(Data *pstate, ExecutionType execType, Type type)
{
    unsigned Lr = prState->GetStateLength();
    unsigned Lc = pcState->GetStateLength();

    // Get Samples for AEM
    SampleStates(type);
    int N = GetSampleLength(Lr);

    // Copy Original Reduced State
    Pointer<double> auxC = pcState->SwapStatePointer(Pointer<double>());
    Pointer<double> auxR = prState->SwapStatePointer(Pointer<double>());
    for (int i = 0; i < N; i++)
    {
        prState->SwapStatePointer(samplesState + Lr * i);
        pcState->SwapStatePointer(samplesState + 2 * Lr * N + Lc * i);
        // Generate Complete State
        R2C(prState, pcState);

        // AEM Sampled States Evolution
        reducedModel->Evolve(prState, ExecutionType::State, type);
        completeModel->Evolve(pcState, ExecutionType::State, type);

        // Retrive Reduced State from Complete State
        prState->SwapStatePointer(samplesState + Lr * N + Lr * i);
        C2R(pcState, prState);
    }
    pcState->SwapStatePointer(auxC);
    prState->SwapStatePointer(auxR);

    // Get Results from AEM
    Math::Iterate(Math::Sub, samplesState, samplesState, Lr, N, Lr, Lr, Lr * N, 0, type);
    Math::Mean(errorState, samplesState + Lr * N, Lr, N, type);
    Math::MatMulNT(0.0, covarState, 1.0 / (N - 1.0), samplesState + Lr * N, samplesState + Lr * N, Lr, N, Lr, type);

    //PrintMatrix("EvolveError", errorState, Lr, 1, type);
    //PrintMatrix("EvolveCovar", covarState, Lr, Lr, type);

    return reducedModel->Evolve(pstate, execType, type);
}

Pointer<double> AEM::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    unsigned Lrs = prState->GetStateLength();
    unsigned Lrm = prMeasure->GetMeasureLength();
    unsigned Lcs = pcState->GetStateLength();
    unsigned Lcm = pcMeasure->GetMeasureLength();

    int N = GetSampleLength(Lrs);

    // Copy Original Reduced State
    Pointer<double> auxCM = pcMeasure->SwapMeasurePointer(Pointer<double>());
    Pointer<double> auxCS = pcState->SwapStatePointer(Pointer<double>());
    Pointer<double> auxRM = prMeasure->SwapMeasurePointer(Pointer<double>());
    Pointer<double> auxRS = prState->SwapStatePointer(Pointer<double>());

    for (int i = 0; i < N; i++)
    {
        prState->SwapStatePointer(samplesState + Lrs * i);
        prMeasure->SwapMeasurePointer(samplesMeasure + Lrm * i);
        pcState->SwapStatePointer(samplesState + 2 * Lrs * N + Lcs * i);
        pcMeasure->SwapMeasurePointer(samplesMeasure + 2 * Lrm * N + Lcm * i);

        // AEM Sampled State Evaluation
        reducedModel->Evaluate(prMeasure, prState, ExecutionType::State, type);
        completeModel->Evaluate(pcMeasure, pcState, ExecutionType::State, type);

        // Retrive Reduced State from Complete State
        prMeasure->SwapMeasurePointer(samplesMeasure + Lrm * N + Lrm * i);
        C2R(pcMeasure, prMeasure);
    }
    prState->SwapStatePointer(auxRS);
    prMeasure->SwapMeasurePointer(auxRM);
    pcState->SwapStatePointer(auxCS);
    pcMeasure->SwapMeasurePointer(auxCM);

    // Get Results from AEM
    Math::Iterate(Math::Sub, samplesMeasure, samplesMeasure, Lrm, N, Lrm, Lrm, Lrm * N, 0, type);
    Math::Mean(errorMeasure, samplesMeasure + Lrm * N, Lrm, N, type);
    Math::MatMulNT(0.0, covarMeasure, 1.0 / (N - 1.0), samplesMeasure + Lrm * N, samplesMeasure + Lrm * N, Lrm, N, Lrm, type);

    //PrintMatrix("EvaluateError", errorMeasure, Lrm, 1, type);
    //PrintMatrix("EvaluateCovar", covarMeasure, Lrm, Lrm, type);
    it++;

    return reducedModel->Evaluate(pmeasure, pstate, execType, type);
}

Data *AEM::GenerateData()
{
    prState = reducedModel->GenerateData();
    pcState = completeModel->GenerateData();

    return prState;
}

Measure *AEM::GenerateMeasure()
{
    prMeasure = reducedModel->GenerateMeasure();
    pcMeasure = completeModel->GenerateMeasure();

    return prMeasure;
}