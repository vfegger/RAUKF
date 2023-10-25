#include "../include/lmodel.hpp"

Pointer<double> LModel::Evolve(Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::Matrix:
        if (type == Type::CPU)
        {
            this->EvolveMatrixCPU(pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvolveMatrixGPU(pstate);
        }
    default:
        throw std::logic_error("LModel Evolve: Execution Type not defined.");
        break;
    }
}
Pointer<double> LModel::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::Matrix:
        if (type == Type::CPU)
        {
            this->EvaluateMatrixCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvaluateMatrixGPU(pmeasure, pstate);
        }
    default:
        throw std::logic_error("LModel Evaluate: Execution Type not defined.");
        break;
    }
}