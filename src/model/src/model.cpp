#include "../include/model.hpp"

void Model::Evolve(Data* pstate, Type type) {
    if (type == Type::CPU)
    {
        Model::EvolveCPU(pstate);
    }
    else if (type == Type::GPU)
    {
        Model::EvolveGPU(pstate);
    }
}

void Model::Evaluate(Measure* pmeasure, Data* pstate, Type type) {
    if (type == Type::CPU)
    {
        Model::EvaluateCPU(pmeasure, pstate);
    }
    else if (type == Type::GPU)
    {
        Model::EvaluateGPU(pmeasure, pstate);
    }
}