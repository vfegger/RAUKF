#include "../include/lmodel.hpp"


void LinearModel::Evolution(Pointer<double> m_o, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvolutionCPU(m_o, pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvolutionGPU(m_o, pstate);
    }
}
void LinearModel::Evaluation(Pointer<double> m_o, Measure *pmeasure, Data *pstate, Type type)
{
    if (type == Type::CPU)
    {
        this->EvaluationCPU(m_o, pmeasure, pstate);
    }
    else if (type == Type::GPU)
    {
        this->EvaluationGPU(m_o, pmeasure, pstate);
    }
}