#ifndef MEASURE_HEADER
#define MEASURE_HEADER

#include "pointer.hpp"
#include <iostream>
#include <string>
#include <map>

struct MeasureInfo
{
    int length;
    double *data;
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
    Pointer<double> instances;
    double **dataOffset;
    double **offset;
    double **covarianceOffset;
    int *lengthPerOffset;
    int length;
    int count;

protected:
public:
    Measure(std::map<std::string, MeasureInfo> &info);
    Pointer<double> GetMeasurePointer();
    Pointer<double> GetMeasureCovariancePointer();
    Pointer<double> GetInstances();
    int GetMeasureLength();
    void SetMeasureData(std::string name, double *data);
    Pointer<double> GetMeasureData();
};

class MeasureLoader
{
private:
    std::map<std::string, MeasureInfo> info;

protected:
public:
    MeasureLoader();
    void Add(std::string name, int length);
    void Link(std::string name, double *data);
    void Remove(std::string name);
    Measure Load();
};

#endif