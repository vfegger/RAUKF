#ifndef STATISTICS_HEADER
#define STATISTICS_HEADER

#include <boost/math/distributions/chi_squared.hpp>

class Statistics
{
private:
protected:
public:
    Statistics();
    ~Statistics();
    double GetChi2(double sigma, int degree);
};

#endif