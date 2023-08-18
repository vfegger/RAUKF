#ifndef HFE_HEADER
#define HFE_HEADER

#include "../../model/include/model.hpp"
#include <random>

class HFE : public Model
{
private:
    int Lx, Ly, Lz;
    double Sx, Sy, Sz;
    double dx, dy, dz;
    double amp;

protected:
    void EvolveCPU(Data *pstate) override;
    void EvolveGPU(Data *pstate) override;
    void EvaluateCPU(Measure *pmeasure, Data *pstate) override;
    void EvaluateGPU(Measure *pmeasure, Data *pstate) override;

    Data &GenerateData() override;
    Measure &GenerateMeasure() override;
public:
};

#endif