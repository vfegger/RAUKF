#include "../include/timer.hpp"

Timer::Timer() : clock()
{
    startTime = Time::now();
    length = 0u;
    duration = NULL;
    count = 0u;
    isValid = false;
}

Timer::Timer(unsigned length_in) : clock()
{
    startTime = Time::now();
    length = length_in;
    if (length_in == 0u)
    {
        std::cout << "Timer: Invalid lenght.\n";
        return;
    }
    duration = new (std::nothrow) double[length_in];
    count = 0u;
    isValid = false;
}

Timer::Timer(const Timer &timer_in) : clock()
{
    startTime = timer_in.startTime;
    length = timer_in.length;
    if (length > 0)
    {
        duration = new (std::nothrow) double[length];
        for (unsigned i = 0u; i < length; i++)
        {
            duration[i] = timer_in.duration[i];
        }
    }
    else
    {
        duration = NULL;
    }
    count = timer_in.count;
    isValid = timer_in.isValid;
}

Timer &Timer::operator=(const Timer &timer_in)
{
    startTime = timer_in.startTime;
    length = timer_in.length;
    if (length > 0)
    {
        duration = new (std::nothrow) double[length];
        for (unsigned i = 0u; i < length; i++)
        {
            duration[i] = timer_in.duration[i];
        }
    }
    else
    {
        duration = NULL;
    }
    count = timer_in.count;
    isValid = timer_in.isValid;
    return *this;
}

void Timer::Start()
{
    startTime = Time::now();
    isValid = false;
}
void Timer::Save()
{
    if (count >= length)
    {
        std::cout << "Error: Insufficient size\n";
    }
    duration[count] = (Time::now() - startTime).count();
    count++;
}
void Timer::Reset()
{
    startTime = Time::now();
    count = 0u;
    isValid = false;
}
void Timer::Print()
{
    if (!isValid)
    {
        SetValues();
    }
    for (unsigned i = 0u; i < count; i++)
    {
        std::cout << "Time duration " << i << ": " << duration[i] << " ms\n";
    }
}
void Timer::SetValues()
{
    if (isValid)
    {
        return;
    }
    for (unsigned i = count - 1u; i > 0u; i--)
    {
        duration[i] -= duration[i - 1u];
    }
    for (unsigned i = 0u; i < count; i++)
    {
        duration[i] /= 1e+6;
    }
    isValid = true;
}
double *Timer::GetValues()
{
    return duration;
}
unsigned Timer::GetCount()
{
    return count;
}
Timer::~Timer()
{
    delete[] duration;
}