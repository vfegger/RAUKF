#ifndef HFE_HEADER
#define HFE_HEADER

#include "../../model/include/aem.hpp"
#include "../../model/include/nlmodel.hpp"
#include "../../model/include/lmodel.hpp"
#include "../include/hc.hpp"

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
    void SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Measure *GenerateMeasure() override;
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
    void SetParms(int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Measure *GenerateMeasure() override;
};

#endif