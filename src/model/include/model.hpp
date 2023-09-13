#ifndef MODEL_HEADER
#define MODEL_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"

class Model
{
private:
protected:
    DataLoader dataLoader;
    MeasureLoader measureLoader;

    virtual void EvolveCPU(Data *pstate, int index) = 0;
    virtual void EvolveGPU(Data *pstate, int index) = 0;
    virtual void EvaluateCPU(Measure *pmeasure, Data *pstate, int index) = 0;
    virtual void EvaluateGPU(Measure *pmeasure, Data *pstate, int index) = 0;

public:
    void EvolveState(Data *pstate, Type type);
    void EvaluateState(Measure *pmeasure, Data *pstate, Type type);
    void Evolve(Data *pstate, Type type);
    void Evaluate(Measure *pmeasure, Data *pstate, Type type);

    virtual Data *GenerateData() = 0;
    virtual Measure *GenerateMeasure() = 0;
};

class LinearModel
{
private:
protected:
    virtual void EvolutionMatrixCPU(Pointer<double> m_o, Data *pstate) = 0;
    virtual void EvolutionMatrixGPU(Pointer<double> m_o, Data *pstate) = 0;
    virtual void EvaluationMatrixCPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate) = 0;
    virtual void EvaluationMatrixGPU(Pointer<double> m_o, Measure *pmeasure, Data *pstate) = 0;

public:
    void EvolutionMatrix(Pointer<double> m_o, Data *pstate, Type type);
    void EvaluationMatrix(Pointer<double> m_o, Measure *pmeasure, Data *pstate, Type type);
};

#endif