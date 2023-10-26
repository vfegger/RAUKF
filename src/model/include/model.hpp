#ifndef MODEL_HEADER
#define MODEL_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"

enum ExecutionType
{
    State,
    Instance,
    Matrix
};

class Model
{
private:
protected:
    DataLoader dataLoader;
    MeasureLoader measureLoader;

public:
    virtual Pointer<double> Evolve(Data *pstate, ExecutionType execType, Type type) = 0;
    virtual Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) = 0;

    virtual void CorrectEstimation(Data *pstate, Type type);
    virtual void CorrectEvaluation(Measure *pmeasure, Data *pstate, Type type);

    virtual Data *GenerateData() = 0;
    virtual Measure *GenerateMeasure() = 0;
};

#endif