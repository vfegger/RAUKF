#include "../include/aem.hpp"

void AEM::SampleStates(Type type)
{
    unsigned Lrs = prState->GetStateLength();
    unsigned Lrm = prMeasure->GetMeasureLength();
    unsigned Lcs = pcState->GetStateLength();
    unsigned Lcm = pcMeasure->GetMeasureLength();
    unsigned Lrs2 = Lrs * Lrs;
    unsigned Ls = (2 * Lrs + Lcs);
    unsigned Lm = (2 * Lrm + Lcm);
    unsigned N = 2 * Lrs + 1;
    unsigned NLs = Ls * N;
    unsigned NLm = Lm * N;
    samplesState.alloc(NLs);
    samplesMeasure.alloc(NLm);
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
    NSamples = N;
}

void AEM::CorrectEstimation(Data *pstate, Type type)
{
    int L = pstate->GetStateLength();
    Math::Add(pstate->GetStatePointer(), errorState, L, type);
    Math::Add(pstate->GetStateCovariancePointer(), covarState, L * L, type);
}

void AEM::CorrectEvaluation(Measure *pmeasure, Data *pstate, Type type)
{
    int L = pmeasure->GetMeasureLength();
    Math::Add(pmeasure->GetMeasurePointer(), errorMeasure, L, type);
    Math::Add(pmeasure->GetMeasureCovariancePointer(), covarMeasure, L * L, type);
}

Pointer<double> AEM::Evolve(Data *pstate, ExecutionType execType, Type type)
{
    unsigned Lr = prState->GetStateLength();
    unsigned Lc = pcState->GetStateLength();

    // Get Samples for AEM
    SampleStates(type);

    // Copy Original Reduced State
    Pointer<double> auxC = pcState->SwapStatePointer(Pointer<double>());
    Pointer<double> auxR = prState->SwapStatePointer(Pointer<double>());
    for (int i = 0; i < NSamples; i++)
    {
        prState->SwapStatePointer(samplesState + Lr * i);
        pcState->SwapStatePointer(samplesState + 2 * Lr * NSamples + Lc * i);
        // Generate Complete State
        R2C(prState, pcState);

        // AEM Sampled States Evolution
        reducedModel->Evolve(prState, ExecutionType::State, type);
        completeModel->Evolve(pcState, ExecutionType::State, type);

        // Retrive Reduced State from Complete State
        prState->SwapStatePointer(samplesState + Lr * NSamples + Lr * i);
        C2R(pcState, prState);
    }
    pcState->SwapStatePointer(auxC);
    prState->SwapStatePointer(auxR);

    // Get Results from AEM
    Math::Iterate(Math::Sub, samplesState, samplesState, Lr, NSamples, Lr, Lr, Lr * NSamples, 0, type);
    Math::Mean(errorState, samplesState + Lr * NSamples, Lr, NSamples, type);
    Math::MatMulNT(0.0, covarState, 1.0, samplesState + Lr * NSamples, samplesState + Lr * NSamples, Lr, NSamples, Lr, type);

    return reducedModel->Evolve(pstate, execType, type);
}

Pointer<double> AEM::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    unsigned Lrs = prState->GetStateLength();
    unsigned Lrm = prMeasure->GetMeasureLength();
    unsigned Lcs = pcState->GetStateLength();
    unsigned Lcm = pcMeasure->GetMeasureLength();

    // Copy Original Reduced State
    Pointer<double> auxCM = pcMeasure->SwapMeasurePointer(Pointer<double>());
    Pointer<double> auxCS = pcState->SwapStatePointer(Pointer<double>());
    Pointer<double> auxRM = prMeasure->SwapMeasurePointer(Pointer<double>());
    Pointer<double> auxRS = prState->SwapStatePointer(Pointer<double>());

    for (int i = 0; i < NSamples; i++)
    {
        prState->SwapStatePointer(samplesState + Lrs * i);
        prMeasure->SwapMeasurePointer(samplesMeasure + Lrm * i);
        pcState->SwapStatePointer(samplesState + 2 * Lrs * NSamples + Lcs * i);
        pcMeasure->SwapMeasurePointer(samplesMeasure + 2 * Lrm * NSamples + Lcm * i);

        // AEM Sampled State Evaluation
        reducedModel->Evaluate(prMeasure, prState, ExecutionType::State, type);
        completeModel->Evaluate(pcMeasure, pcState, ExecutionType::State, type);

        // Retrive Reduced State from Complete State
        prMeasure->SwapMeasurePointer(samplesMeasure + Lrm * NSamples + Lrm * i);
        C2R(pcMeasure, prMeasure);
    }
    prState->SwapStatePointer(auxRS);
    prMeasure->SwapMeasurePointer(auxRM);
    pcState->SwapStatePointer(auxCS);
    pcMeasure->SwapMeasurePointer(auxCM);

    // Get Results from AEM
    Math::Iterate(Math::Sub, samplesMeasure, samplesMeasure, Lrm, NSamples, Lrm, Lrm, Lrm * NSamples, 0, type);
    Math::Mean(errorState, samplesMeasure + Lrm * NSamples, Lrm, NSamples, type);
    Math::MatMulNT(0.0, covarState, 1.0, samplesMeasure + Lrm * NSamples, samplesMeasure + Lrm * NSamples, Lrm, NSamples, Lrm, type);

    return reducedModel->Evaluate(pmeasure, pstate, execType, type);
}