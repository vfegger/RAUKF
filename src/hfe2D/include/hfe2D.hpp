#ifndef HFE2D_HEADER
#define HFE2D_HEADER

#include "../../model/include/lmodel.hpp"
#include "../include/hc2D.hpp"

#define FORWARD_METHOD 1

class HFE2D : public LModel
{
private:
    HC2D::HCParms parms;

    Pointer<double> workspaceState;
    Pointer<double> workspaceMeasure;
    bool isValidState;
    bool isValidMeasure;

protected:
    Pointer<double> EvolveMatrixCPU(Data *pstate, Control *pcontrol) override;
    Pointer<double> EvolveMatrixGPU(Data *pstate, Control *pcontrol) override;
    Pointer<double> EvaluateMatrixCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateMatrixGPU(Measure *pmeasure, Data *pstate) override;

    Pointer<double> EvolveStateCPU(Data *pstate, Control *pcontrol) override;
    Pointer<double> EvolveStateGPU(Data *pstate, Control *pcontrol) override;
    Pointer<double> EvaluateStateCPU(Measure *pmeasure, Data *pstate) override;
    Pointer<double> EvaluateStateGPU(Measure *pmeasure, Data *pstate) override;

public:
    void SetParms(int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double h);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Control *GenerateControl() override;
    Measure *GenerateMeasure() override;
};
#endif