#ifndef KF_HEADER
#define KF_HEADER

#include "../../structure/include/pointer.hpp"
#include "../../structure/include/data.hpp"
#include "../../structure/include/measure.hpp"
#include "../../math/include/math.hpp"
#include "../../model/include/lmodel.hpp"
#include "../../timer/include/timer.hpp"
#include "../../statistics/include/statistics.hpp"

class KF
{
private:
    Statistics *pstatistics;
    LinearModel *pmodel;
    Data *pstate;
    Measure *pmeasure;

    Type type;

protected:
public:
    KF();
    ~KF();
    
    void SetModel(LinearModel *pmodel);
    void UnsetModel();
    void SetType(Type type);
    
    void SetMeasure(std::string name, double *data);
    void GetState(std::string name, double *data);
    void GetStateCovariance(std::string name, double *data);

    void Iterate(Timer &timer);
};

#endif