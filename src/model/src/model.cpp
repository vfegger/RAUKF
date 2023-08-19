#include "../include/model.hpp"

void Model::Evolve(Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvolveCPU(pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvolveGPU(pstate);
    }
}

void Model::Evaluate(Measure *pmeasure, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvaluateCPU(pmeasure, pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvaluateGPU(pmeasure, pstate);
    }
}