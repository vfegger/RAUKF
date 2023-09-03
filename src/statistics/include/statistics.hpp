#ifndef STATISTICS_HEADER
#define STATISTICS_HEADER

#include <julia.h>

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