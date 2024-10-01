#include "../include/statistics.hpp"

Statistics::Statistics()
{
}
Statistics::~Statistics()
{
}
double Statistics::GetChi2(double sigma, int degree)
{
    // Create a chi-squared distribution with the given degree of freedom
    boost::math::chi_squared_distribution<> chi2_dist(degree);

    // Calculate the critical value for the given sigma (confidence level)
    double critical_value = boost::math::quantile(chi2_dist, sigma);

    return critical_value;
}