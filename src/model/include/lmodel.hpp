#ifndef LMODEL_HEADER
#define LMODEL_HEADER

#include "model.hpp"

class LModel : public Model
{
private:
protected:
    virtual Pointer<double> EvolveMatrixCPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvolveMatrixGPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvaluateMatrixCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual Pointer<double> EvaluateMatrixGPU(Measure *pmeasure, Data *pstate) = 0;

    virtual Pointer<double> EvolveStateCPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvolveStateGPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvaluateStateCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual Pointer<double> EvaluateStateGPU(Measure *pmeasure, Data *pstate) = 0;

public:
    virtual Pointer<double> Evolve(Data *pstate, Control *pcontrol, ExecutionType execType, Type type) override;
    virtual Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) override;
};

#endif