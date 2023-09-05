#ifndef RAUKF_HEADER
#define RAUKF_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"
#include "../../math/include/math.hpp"
#include "../../model/include/model.hpp"
#include "../../timer/include/timer.hpp"
#include "../../statistics/include/statistics.hpp"

class RAUKF
{
private:
    Statistics *pstatistics;
    Model *pmodel;
    Data *pstate;
    Measure *pmeasure;

    double alpha, beta, kappa;
    double lambda;
    Pointer<double> wm, wc;

    Type type;

protected:
    void UnscentedTransformation(Pointer<double> xs_o, Pointer<double> x_i, Pointer<double> P_i, int Lx, int Ls, Pointer<double> workspace, Type type);
public:
    RAUKF();
    ~RAUKF();

    void SetParameters(double alpha, double beta, double kappa);
    void SetModel(Model *pmodel);
    void UnsetModel();
    void SetType(Type type);
    void SetWeight();
    void UnsetWeight();
    void SetMeasure(std::string name, double *data);
    void GetState(std::string name, double *data);
    void GetStateCovariance(std::string name, double *data);

    void Iterate(Timer &timer);
};

#endif