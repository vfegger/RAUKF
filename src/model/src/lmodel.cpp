#include "../include/lmodel.hpp"

Pointer<double> LModel::Evolve(Data *pstate, Control *pcontrol, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::Matrix:
        if (type == Type::CPU)
        {
            return this->EvolveMatrixCPU(pstate, pcontrol);
        }
        else if (type == Type::GPU)
        {
            return this->EvolveMatrixGPU(pstate, pcontrol);
        }
    case ExecutionType::State:
        if (type == Type::CPU)
        {
            return this->EvolveStateCPU(pstate, pcontrol);
        }
        else if (type == Type::GPU)
        {
            return this->EvolveStateGPU(pstate, pcontrol);
        }
        break;
    default:
        throw std::logic_error("LModel Evolve: Execution Type not defined.");
        break;
    }
    return Pointer<double>();
}
Pointer<double> LModel::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::Matrix:
        if (type == Type::CPU)
        {
            return this->EvaluateMatrixCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            return this->EvaluateMatrixGPU(pmeasure, pstate);
        }
    case ExecutionType::State:
        if (type == Type::CPU)
        {
            return this->EvaluateStateCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            return this->EvaluateStateGPU(pmeasure, pstate);
        }
        break;
    default:
        throw std::logic_error("LModel Evaluate: Execution Type not defined.");
        break;
    }
    return Pointer<double>();
}