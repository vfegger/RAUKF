#ifndef LMODEL_HEADER
#define LMODEL_HEADER

#include "model.hpp"

class LModel : public Model
{
private:
protected:
    virtual Pointer<double> EvolveMatrixCPU(Data *pstate) = 0;
    virtual Pointer<double> EvolveMatrixGPU(Data *pstate) = 0;
    virtual Pointer<double> EvaluateMatrixCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual Pointer<double> EvaluateMatrixGPU(Measure *pmeasure, Data *pstate) = 0;
public:

    virtual Pointer<double> Evolve(Data *pstate, ExecutionType execType, Type type) override;
    virtual Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) override;
};

#endif