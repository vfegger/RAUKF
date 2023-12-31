#ifndef MEASURE_HEADER
#define MEASURE_HEADER

#include "pointer.hpp"
#include <iostream>
#include <string>
#include <map>

struct MeasureInfo
{
    int length;
    double *noiseData;
    bool linked;

    MeasureInfo();
    MeasureInfo(int length);
};

struct Measure
{
private:
    std::map<std::string, int> index;
    Pointer<double> data;
    Pointer<double> pointer;
    Pointer<double> covariancePointer;
    Pointer<double> noisePointer;
    Pointer<double> instances;
    int *offset;
    int *offset2;
    int *lengthPerOffset;
    int length;
    int count;

protected:
public:
    Measure(std::map<std::string, MeasureInfo> &info);
    Pointer<double> GetMeasurePointer();
    Pointer<double> GetMeasureCovariancePointer();
    Pointer<double> GetMeasureNoisePointer();
    Pointer<double> GetInstances();
    void SetInstances(int length);
    void UnsetInstances();
    int GetMeasureLength();
    int GetOffset(std::string name);
    int GetOffset2(std::string name);
    void SetMeasureData(std::string name, double *data);
    Pointer<double> GetMeasureData();
    void GetMeasureData(std::string name, double *data);
    void GetMeasureCovarianceData(std::string name, double *data);
    
    // Utils for memory manipulation
    Pointer<double> SwapMeasurePointer(Pointer<double> pmeasure);
};

class MeasureLoader
{
private:
    std::map<std::string, MeasureInfo> info;

protected:
public:
    MeasureLoader();
    void Add(std::string name, int length);
    void Link(std::string name, double *noiseData);
    void Remove(std::string name);
    Measure *Load();
};

#endif