#include "../include/nlmodel.hpp"

Pointer<double> NLModel::Evolve(Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::State:
        if (type == Type::CPU)
        {
            this->EvolveStateCPU(pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvolveStateGPU(pstate);
        }
        break;
    case ExecutionType::Instance:
        if (type == Type::CPU)
        {
            this->EvolveInstanceCPU(pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvolveInstanceGPU(pstate);
        }
        break;
    default:
        throw std::logic_error("NLModel Evolve: Execution Type not defined.");
        break;
    }
}
Pointer<double> NLModel::Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type)
{
    switch (execType)
    {
    case ExecutionType::State:
        if (type == Type::CPU)
        {
            this->EvaluateStateCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvaluateStateGPU(pmeasure, pstate);
        }
        break;
    case ExecutionType::Instance:
        if (type == Type::CPU)
        {
            this->EvaluateInstanceCPU(pmeasure, pstate);
        }
        else if (type == Type::GPU)
        {
            this->EvaluateInstanceGPU(pmeasure, pstate);
        }
        break;
    default:
        throw std::logic_error("NLModel Evaluate: Execution Type not defined.");
        break;
    }
}