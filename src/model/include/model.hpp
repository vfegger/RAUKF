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

    virtual void EvolveCPU(Data *pstate) = 0;
    virtual void EvolveGPU(Data *pstate) = 0;
    virtual void EvaluateCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual void EvaluateGPU(Measure *pmeasure, Data *pstate) = 0;

    virtual Data &GenerateData() = 0;
    virtual Measure &GenerateMeasure() = 0;

public:
    void Evolve(Data *pstate, Type type);
    void Evaluate(Measure *pmeasure, Data *pstate, Type type);
};

#endif