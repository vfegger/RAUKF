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
    Control *prControl;
    Control *pcControl;
    Measure *prMeasure;
    Measure *pcMeasure;

    Pointer<double> samplesState;
    Pointer<double> samplesMeasure;

    // Need to allocate samples with the following data: [Reduced State] + [State Error] + [Complete State] and [Reduced Measure] + [Measure Error] + [Complete Measure]
    void SetMemory(int Lcs, int Lcm, int Lrs, int Lrm, Type type);
    void UnsetMemory(Type type);
    virtual int GetSampleLength(int Lrs);
    virtual void SampleStates(Type type);
    void SetModel(Model *rm, Model *cm);

public:
    virtual void R2C(Data *prState, Data *pcState) = 0;
    virtual void R2C(Measure *prMeasure, Measure *pcMeasure) = 0;
    virtual void C2R(Data *pcState, Data *prState) = 0;
    virtual void C2R(Measure *pcMeasure, Measure *prMeasure) = 0;

    void CorrectEstimation(Data *pstate, Type type) override;
    void CorrectEvaluation(Measure *pmeasure, Data *pstate, Type type) override;

    Pointer<double> Evolve(Data *pstate, Control *pcontrol, ExecutionType execType, Type type) override;
    Pointer<double> Evaluate(Measure *pmeasure, Data *pstate, ExecutionType execType, Type type) override;

    Data *GenerateData() override;
    Control *GenerateControl() override;
    Measure *GenerateMeasure() override;
};

#endif