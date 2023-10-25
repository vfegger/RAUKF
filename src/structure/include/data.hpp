#ifndef DATA_HEADER
#define DATA_HEADER

#include "pointer.hpp"
#include <iostream>
#include <string>
#include <map>

struct DataInfo
{
    int length;
    double *data;
    double *covarianceData;
    double *noiseData;
    bool linked;

    DataInfo();
    DataInfo(int length);
};

struct Data
{
private:
    std::map<std::string, int> index;
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
    Data(std::map<std::string, DataInfo> &info);
    Pointer<double> GetStatePointer();
    Pointer<double> GetStateCovariancePointer();
    Pointer<double> GetStateNoisePointer();
    Pointer<double> GetInstances();
    void GetStateData(std::string name, double *data);
    void GetStateCovarianceData(std::string name, double *data);
    void SetInstances();
    void UnsetInstances();
    int GetStateLength();
    int GetSigmaLength();
    int GetOffset(std::string name);
    int GetOffset2(std::string name);

    // Utils for memory manipulation
    Pointer<double> SwapStatePointer(Pointer<double> pstate);
};

class DataLoader
{
private:
    std::map<std::string, DataInfo> info;

protected:
public:
    DataLoader();
    void Add(std::string name, int length);
    void Link(std::string name, double *data, double *covarianceData, double *noiseData);
    void Remove(std::string name);
    Data *Load();
};

#endif