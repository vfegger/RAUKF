#ifndef NLMODEL_HEADER
#define NLMODEL_HEADER

#include "model.hpp"

class NLModel : public Model
{
private:
protected:
    virtual Pointer<double> EvolveStateCPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvolveStateGPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvaluateStateCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual Pointer<double> EvaluateStateGPU(Measure *pmeasure, Data *pstate) = 0;

    virtual Pointer<double> EvolveInstanceCPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvolveInstanceGPU(Data *pstate, Control *pcontrol) = 0;
    virtual Pointer<double> EvaluateInstanceCPU(Measure *pmeasure, Data *pstate) = 0;
    virtual Pointer<double> EvaluateInstanceGPU(Measure *pmeasure, Data *pstate) = 0;

public:
    virtual Pointer<double> Evolve(Data *pstate, Control *pcontrol, ExecutionType execType, Type type) override;
    virtual Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) override;
};

#endif