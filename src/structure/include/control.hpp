#ifndef CONTROL_HEADER
#define CONTROL_HEADER

#include "pointer.hpp"
#include <iostream>
#include <string>
#include <map>

struct ControlInfo
{
    int length;
    double *data;
    double *covarianceData;
    double *noiseData;
    bool linked;

    ControlInfo();
    ControlInfo(int length);
};

struct Control
{
private:
    std::map<std::string, int> index;
    Pointer<double> pointer;
    int *offset;
    int *lengthPerOffset;
    int length;
    int count;

protected:
public:
    Control(std::map<std::string, ControlInfo> &info);
    Pointer<double> GetControlPointer();
    void SetControlData(std::string name, double *data);
    void GetControlData(std::string name, double *data);
    int GetControlLength();
    int GetOffset(std::string name);
};

class ControlLoader
{
private:
    std::map<std::string, ControlInfo> info;

protected:
public:
    ControlLoader();
    void Add(std::string name, int length);
    void Link(std::string name, double *data);
    void Remove(std::string name);
    Control *Load();
};

#endif