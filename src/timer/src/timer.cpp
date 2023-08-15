#include "../include/timer.hpp"

Timer::Timer() : clock() {}

void Timer::Record(Type type)
{
    if (type == Type::CPU)
    {
        timePoints.push_back(clock.now());
    }
    else if (type == Type::GPU)
    {
        cudaEvent_t event;
        cudaEventCreate(&event);
        cudaEvents.push_back(event);
        cudaEventRecord(event);
    }
}

void Timer::Reset(Type type)
{
    if (type == Type::CPU)
    {
        timePoints.clear();
    }
    else if (type == Type::GPU)
    {
        for (cudaEvent_t event : cudaEvents)
        {
            cudaEventDestroy(event);
        }
        cudaEvents.clear();
    }
}

std::list<double> Timer::GetResult(Type type)
{
    std::list<double> results;
    if (type == Type::CPU)
    {
        std::list<TimePoint>::iterator it = timePoints.begin();
        for (std::list<TimePoint>::iterator i = ++timePoints.begin(); i != timePoints.end(); ++i)
        {
            results.push_back((*i - *it).count() / 1e+6);
            it = i;
        }
    }
    else if (type == Type::GPU)
    {
        std::list<cudaEvent_t>::iterator it = cudaEvents.begin();
        for (std::list<cudaEvent_t>::iterator i = ++cudaEvents.begin(); i != cudaEvents.end(); ++i)
        {
            float time = 0.0f;
            cudaEventSynchronize(*i);
            cudaEventElapsedTime(&time, *it, *i);
            results.push_back((double)time);
            it = i;
        }
    }
    return results;
}