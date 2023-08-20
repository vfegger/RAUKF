#ifndef RAUKF_HEADER
#define RAUKF_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"
#include "../../math/include/math.hpp"
#include "../../model/include/model.hpp"
#include "../../timer/include/timer.hpp"

class RAUKF
{
private:
    Model *pmodel;
    Data *pstate;
    Measure *pmeasure;

    double alpha, beta, kappa;

    Type type;

protected:
public:
    RAUKF();

    void SetParameters(double alpha, double beta, double kappa);
    void SetModel(Model *pmodel);
    void UnsetModel();
    void SetType(Type type);

    void Iterate(Timer &timer);
};

#endif