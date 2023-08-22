#include "../include/measure.hpp"

MeasureLoader::MeasureLoader() {}

MeasureInfo::MeasureInfo() : length(0), linked(false) {}

MeasureInfo::MeasureInfo(int length) : length(length), linked(false) {}

Measure::Measure(std::map<std::string, MeasureInfo> &info) : count(info.size()), length(0)
{
    if (count == 0)
    {
        return;
    }
    offset = (int *)malloc(sizeof(int) * count);
    offset2 = (int *)malloc(sizeof(int) * count);
    lengthPerOffset = (int *)malloc(sizeof(int) * count);
    int iaux = 0;
    for (std::map<std::string, MeasureInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        index[(*i).first] = iaux;
        int l = (*i).second.length;
        lengthPerOffset[iaux] = l;
        length += l;
        ++iaux;
    }
    offset[0] = 0;
    offset2[0] = 0;
    for (int i = 1; i < count; ++i)
    {
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
        offset2[i] = offset2[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
    }
    data.alloc(length);
    pointer.alloc(length);
    covariancePointer.alloc(length * length);
    noisePointer.alloc(length * length);
    double *pdHost = data.host();
    double *pHost = pointer.host();
    double *pcHost = covariancePointer.host();
    double *pnHost = noisePointer.host();
    for (int j = 0; j < length; ++j)
    {
        pdHost[j] = 0.0;
        pHost[j] = 0.0;
        for (int i = 0; i < length; ++i)
        {
            pcHost[j * length + i] = 0.0;
            pnHost[j * length + i] = 0.0;
        }
    }
    for (std::map<std::string, MeasureInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            double *aux0 = pdHost + offset[ii];
            double *aux1 = pHost + offset[ii];
            double *aux2 = pcHost + offset2[ii];
            double *aux3 = pnHost + offset2[ii];
            for (int j = 0; j < lengthPerOffset[ii]; ++j)
            {
                aux3[j * length + j] = (*i).second.noiseData[j];
            }
        }
        else
        {
            std::cout << "Link failed in data of name: " + (*i).first + ".\n";
        }
    }
    noisePointer.copyHost2Dev(length * length);
    instances.alloc((2 * length + 1) * length);
}

Pointer<double> Measure::GetMeasurePointer()
{
    return pointer;
}

Pointer<double> Measure::GetMeasureCovariancePointer()
{
    return covariancePointer;
}

Pointer<double> Measure::GetMeasureNoisePointer()
{
    return noisePointer;
}

Pointer<double> Measure::GetInstances()
{
    return instances;
}

void Measure::SetInstances(int stateLength)
{
    instances.alloc((2 * stateLength + 1) * length);
}

void Measure::UnsetInstances()
{
    instances.free();
}

int Measure::GetMeasureLength()
{
    return length;
}

int Measure::GetOffset(std::string name)
{
    return offset[index[name]];
}

int Measure::GetOffset2(std::string name)
{
    return offset2[index[name]];
}

void Measure::SetMeasureData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = (this->data.host() + offset[ii]);
    int l = lengthPerOffset[ii];
    for (int i = 0; i < l; ++i)
    {
        pHost[i] = data[i];
    }
}

Pointer<double> Measure::GetMeasureData()
{
    data.copyHost2Dev(length);
    return data;
}

void MeasureLoader::Add(std::string name, int length)
{
    info[name] = MeasureInfo(length);
}

void MeasureLoader::Link(std::string name, double *noiseData)
{
    info[name].noiseData = noiseData;
    info[name].linked = true;
}

void MeasureLoader::Remove(std::string name)
{
    info.erase(name);
}

Measure *MeasureLoader::Load()
{
    return new Measure(info);
}