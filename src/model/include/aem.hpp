#ifndef AEM_HEADER
#define AEM_HEADER

#include "model.hpp"
#include "lmodel.hpp"

class AEM
{
private:
protected:
    LinearModel *reducedModel;
    Model *completeModel;

    virtual void SampleStates(Data *pstate);
public:
    virtual void R2C(Data *pcState, Measure *pcMeasure, Data *prState, Measure *prMeasure) = 0;
    virtual void C2R(Data *pcState, Measure *pcMeasure, Data *prState, Measure *prMeasure) = 0;

    
};

#endif