#ifndef KF_HEADER
#define KF_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"
#include "../../structure/include/control.hpp"
#include "../../math/include/math.hpp"
#include "../../model/include/model.hpp"
#include "../../timer/include/timer.hpp"
#include "../../statistics/include/statistics.hpp"

class KF
{
private:
    Statistics *pstatistics;
    Model *pmodel;
    Data *pstate;
    Measure *pmeasure;
    Control *pcontrol;

    Type type;

protected:
    void PrintMatrix(std::string name, double *mat, int lengthI, int lengthJ);
    void PrintMatrix(std::string name, Pointer<double> mat, int lengthI, int lengthJ, Type type);

public:
    KF();
    ~KF();

    void SetModel(Model *pmodel);
    void UnsetModel();
    void SetType(Type type);

    void SetMeasure(std::string name, double *data);
    void GetMeasure(std::string name, double *data);
    void GetMeasureCovariance(std::string name, double *data);
    void GetState(std::string name, double *data);
    void GetStateCovariance(std::string name, double *data);

    void Iterate(Timer &timer);
};

#endif