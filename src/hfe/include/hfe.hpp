#ifndef HFE_HEADER
#define HFE_HEADER

#include "../../model/include/aem.hpp"
#include "../../model/include/nlmodel.hpp"
#include "../../model/include/lmodel.hpp"
#include "../include/hc.hpp"
#include "../../math/include/interpolation.hpp"

#define FORWARD_METHOD 1

class HFE_RM : public LModel
{
private:
    HC::HCParms parms;

    Pointer<double> workspace;
    bool isValidState;
    bool isValidMeasure;

protected:
    Pointer<double> EvolveMatrixCPU(Data *pstate) override;
    Pointer<double> EvolveMatrixGPU(Data *pstate) override;
    Pointer<double> EvaluateMatrixCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateMatrixGPU(Measure *pmeasure, Data *pstate) override;

    Pointer<double> EvolveStateCPU(Data *pstate) override;
    Pointer<double> EvolveStateGPU(Data *pstate) override;
    Pointer<double> EvaluateStateCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateStateGPU(Measure *pmeasure, Data *pstate) override;

public:
    void SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double T_ref);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Measure *GenerateMeasure() override;

    friend class HFE_AEM;
};

class HFE : public NLModel
{
private:
    HC::HCParms parms;

    double *workspace;

protected:
    Pointer<double> EvolveInstanceCPU(Data *pstate) override;
    Pointer<double> EvolveInstanceGPU(Data *pstate) override;
    Pointer<double> EvaluateInstanceCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateInstanceGPU(Measure *pmeasure, Data *pstate) override;

    Pointer<double> EvolveStateCPU(Data *pstate) override;
    Pointer<double> EvolveStateGPU(Data *pstate) override;
    Pointer<double> EvaluateStateCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateStateGPU(Measure *pmeasure, Data *pstate) override;

public:
    void SetParms(int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp, double T_ref = 0);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Measure *GenerateMeasure() override;
    
    friend class HFE_AEM;
};

class HFE_AEM : public AEM
{
private:    
    Pointer<double> workspace;
    Type type;
protected:
public:
    void SetModel(HFE_RM* rm, HFE* cm);
    void SetMemory(Type type);
    void UnsetMemory(Type type);

    void R2C(Data *prState, Data *pcState) override;
    void R2C(Measure *prMeasure, Measure *pcMeasure) override;
    void C2R(Data *pcState, Data *prState) override;
    void C2R(Measure *pcMeasure, Measure *prMeasure) override;

    int GetSampleLength(int Lrs) override;
    void SampleStates(Type type) override;
};
#endif