#ifndef AEM_HEADER
#define AEM_HEADER

#include "model.hpp"
#include "../../math/include/math.hpp"

class AEM : public Model
{
private:
protected:
    Model *reducedModel;
    Model *completeModel;

    Pointer<double> errorState;
    Pointer<double> errorMeasure;
    Pointer<double> covarState;
    Pointer<double> covarMeasure;

    Data *prState;
    Data *pcState;
    Measure *prMeasure;
    Measure *pcMeasure;

    Pointer<double> samplesState;
    Pointer<double> samplesMeasure;
    unsigned NSamples;

    // Need to allocate samples with the following data: [Reduced State] + [State Error] + [Complete State] and [Reduced Measure] + [Measure Error] + [Complete Measure]
    virtual void SampleStates(Type type);

public:
    virtual void R2C(Data *prState, Data *pcState) = 0;
    virtual void R2C(Measure *prMeasure, Measure *pcMeasure) = 0;
    virtual void C2R(Data *pcState, Data *prState) = 0;
    virtual void C2R(Measure *pcMeasure, Measure *prMeasure) = 0;

    virtual Pointer<double> Evolve(Data *pstate, ExecutionType execType, Type type) override;
    virtual Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) override;
};

#endif