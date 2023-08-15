#include "../include/measure.hpp"

MeasureLoader::MeasureLoader() {}

MeasureInfo::MeasureInfo() : length(0), data(NULL), linked(false) {}

MeasureInfo::MeasureInfo(int length) : length(length), data(NULL), linked(false) {}

Measure::Measure(std::map<std::string, MeasureInfo> &info) : count(info.size())
{
    if (count == 0)
    {
        return;
    }
    dataOffset = (double **)malloc(sizeof(double *) * 2 * count);
    offset = (double **)malloc(sizeof(double *) * 2 * count);
    covarianceOffset = (double **)malloc(sizeof(double *) * 2 * count);
    noiseOffset = (double **)malloc(sizeof(double *) * 2 * count);
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
    data.alloc(length);
    pointer.alloc(length);
    covariancePointer.alloc(length * length);
    noisePointer.alloc(length * length);
    double *pdHost = data.host();
    double *pdDev = data.dev();
    double *pHost = pointer.host();
    double *pDev = pointer.dev();
    double *pcHost = covariancePointer.host();
    double *pcDev = covariancePointer.dev();
    double *pnHost = noisePointer.host();
    double *pnDev = noisePointer.dev();
    dataOffset[0] = pdHost;
    dataOffset[count] = pdDev;
    offset[0] = pHost;
    offset[count] = pDev;
    covarianceOffset[0] = pcHost;
    covarianceOffset[count] = pcDev;
    noiseOffset[0] = pnHost;
    noiseOffset[count] = pnDev;
    for (int i = 1; i < count; ++i)
    {
        dataOffset[i] = dataOffset[i - 1] + lengthPerOffset[i - 1];
        dataOffset[count + i] = dataOffset[count + i - 1] + lengthPerOffset[i - 1];
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
        offset[count + i] = offset[count + i - 1] + lengthPerOffset[i - 1];
        covarianceOffset[i] = covarianceOffset[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        covarianceOffset[count + i] = covarianceOffset[count + i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        noiseOffset[i] = noiseOffset[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        noiseOffset[count + i] = noiseOffset[count + i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
    }
    for (std::map<std::string, MeasureInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            for (int j = 0; j < lengthPerOffset[ii]; ++j)
            {
                offset[ii][j] = (*i).second.data[j];
                noiseOffset[ii][j * length + j] = (*i).second.noiseData[j];
            }
        }
        else
        {
            std::cout << "Link failed in data of name: " + (*i).first + ".\n";
        }
    }
    pointer.copyHost2Dev(length);
    noisePointer.copyHost2Dev(length);
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

Pointer<double> Measure::GetInstances()
{
    return instances;
}

int Measure::GetMeasureLength()
{
    return length;
}

void Measure::SetMeasureData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = dataOffset[ii];
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

void MeasureLoader::Link(std::string name, double *data, double *noiseData)
{
    info[name].data = data;
    info[name].noiseData = noiseData;
    info[name].linked = true;
}

void MeasureLoader::Remove(std::string name)
{
    info.erase(name);
}

Measure MeasureLoader::Load()
{
    return Measure(info);
}