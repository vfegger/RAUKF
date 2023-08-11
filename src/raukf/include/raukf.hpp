#ifndef RAUKF_HEADER
#define RAUKF_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../math/include/math.hpp"

class Model;

class RAUKF
{
private:
    Model *pmodel;
    Data *pstate;
    Measure *pmeasure;

    double alpha, beta, kappa;
    double lambda;

    Type type;

protected:
public:
    RAUKF();

    void SetModel(Model *pmodel);
    void SetState(Data *pstate);
    void SetType(Type type);

    void Iterate(Timer &timer);
};

#endif