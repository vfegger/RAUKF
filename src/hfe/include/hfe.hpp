#ifndef HFE_HEADER
#define HFE_HEADER

#include "../../model/include/model.hpp"
#include "../include/hc.hpp"

#define FORWARD_METHOD 1

class HFE : public Model
{
private:
    HC::HCParms parms;

    double *workspace;

protected:
    void EvolveCPU(Data *pstate, int index) override;
    void EvolveGPU(Data *pstate, int index) override;
    void EvaluateCPU(Measure *pmeasure, Data *pstate, int index) override;
    void EvaluateGPU(Measure *pmeasure, Data *pstate, int index) override;

public:
    void SetParms(int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp);
    void SetMemory(Type type);
    void UnsetMemory(Type type);
    Data *GenerateData() override;
    Measure *GenerateMeasure() override;
};

#endif