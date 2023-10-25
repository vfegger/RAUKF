#include "../include/nlmodel.hpp"

Pointer<double> NLModel::Evolve(Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::State:
        if (type == Type::CPU)
        {
            return this->EvolveStateCPU(pstate);
        }
        else if (type == Type::GPU)
        {
            return this->EvolveStateGPU(pstate);
        }
        break;
    case ExecutionType::Instance:
        if (type == Type::CPU)
        {
            return this->EvolveInstanceCPU(pstate);
        }
        else if (type == Type::GPU)
        {
            return this->EvolveInstanceGPU(pstate);
        }
        break;
    default:
        throw std::logic_error("NLModel Evolve: Execution Type not defined.");
        break;
    }
    return Pointer<double>();
}
Pointer<double> NLModel::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
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
    case ExecutionType::Instance:
        if (type == Type::CPU)
        {
            return this->EvaluateInstanceCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            return this->EvaluateInstanceGPU(pmeasure, pstate);
        }
        break;
    default:
        throw std::logic_error("NLModel Evaluate: Execution Type not defined.");
        break;
    }
    return Pointer<double>();
}