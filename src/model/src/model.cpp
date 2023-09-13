#include "../include/model.hpp"

void Model::EvolveState(Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvolveCPU(pstate, -1);
    }
    else if (type == Type::GPU)
    {
        this->EvolveGPU(pstate, -1);
    }
}

void Model::EvaluateState(Measure *pmeasure, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvaluateCPU(pmeasure, pstate, -1);
    }
    else if (type == Type::GPU)
    {
        this->EvaluateGPU(pmeasure, pstate, -1);
    }
}

void Model::Evolve(Data *pstate, Type type)
{
    int Lsigma = pstate->GetSigmaLength();
    if (type == Type::CPU)
    {
        for (int s = 0; s < Lsigma; ++s)
        {
            this->EvolveCPU(pstate, s);
        }
    }
    else if (type == Type::GPU)
    {
        for (int s = 0; s < Lsigma; ++s)
        {
            this->EvolveGPU(pstate, s);
        }
    }
}

void Model::Evaluate(Measure *pmeasure, Data *pstate, Type type)
{
    int Lsigma = pstate->GetSigmaLength();
    if (type == Type::CPU)
    {
        for (int s = 0; s < Lsigma; ++s)
        {
            this->EvaluateCPU(pmeasure, pstate, s);
        }
    }
    else if (type == Type::GPU)
    {
        for (int s = 0; s < Lsigma; ++s)
        {
            this->EvaluateGPU(pmeasure, pstate, s);
        }
    }
}

void LinearModel::EvolutionMatrix(Pointer<double> m_o, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvolutionMatrixCPU(m_o, pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvolutionMatrixGPU(m_o, pstate);
    }
}
void LinearModel::EvaluationMatrix(Pointer<double> m_o, Measure *pmeasure, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvaluationMatrixCPU(m_o, pmeasure, pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvaluationMatrixGPU(m_o, pmeasure, pstate);
    }
}