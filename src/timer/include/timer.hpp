#ifndef TIMER_HEADER
#define TIMER_HEADER

#include <chrono>
#include <list>
#include <cuda_runtime.h>
#include "../../structure/include/pointer.hpp"

using Time = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double>;
using TimePoint = std::chrono::time_point<Time, Duration>;

class Timer
{
private:
    Time clock;
    std::list<TimePoint> timePoints;
    std::list<cudaEvent_t> cudaEvents;

public:
    Timer();

    void Record(Type type);
    void Reset(Type type);
    std::list<double> GetResult(Type type);
};

#endif