#ifndef LMODEL_HEADER
#define LMODEL_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"

class LinearModel
{
private:
protected:
    DataLoader dataLoader;
    MeasureLoader measureLoader;

    virtual void EvolutionCPU(Pointer<double> m_o, Data *pstate) = 0;
    virtual void EvolutionGPU(Pointer<double> m_o, Data *pstate) = 0;
    virtual void EvaluationCPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate) = 0;
    virtual void EvaluationGPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate) = 0;

public:
    void Evolution(Pointer<double> m_o, Data *pstate, Type type);
    void Evaluation(Pointer<double> m_o, Measure *pmeasure, Data *pstate, Type type);

    virtual Data *GenerateLinearData() = 0;
    virtual Measure *GenerateLinearMeasure() = 0;
};

#endif