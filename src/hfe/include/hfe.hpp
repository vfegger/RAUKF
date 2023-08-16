#ifndef HFE_HEADER
#define HFE_HEADER

#include "../../model/include/model.hpp"

class HFE : public Model
{
private:
    int Lx, Ly, Lz;

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