#include "../include/data.hpp"

DataLoader::DataLoader() {}

DataInfo::DataInfo() : length(0), data(NULL), linked(false) {}

DataInfo::DataInfo(int length) : length(length), data(NULL), linked(false) {}

Data::Data(std::map<std::string, DataInfo> &info) : count(info.size()), length(0)
{
    if (count == 0)
    {
        return;
    }
    offset = (double **)malloc(sizeof(double *) * 2 * count);
    covarianceOffset = (double **)malloc(sizeof(double *) * 2 * count);
    noiseOffset = (double **)malloc(sizeof(double *) * 2 * count);
    lengthPerOffset = (int *)malloc(sizeof(int) * count);
    int iaux = 0;
    for (std::map<std::string, DataInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        index[(*i).first] = iaux;
        int l = (*i).second.length;
        lengthPerOffset[iaux] = l;
        length += l;
        ++iaux;
    }
    pointer.alloc(length);
    covariancePointer.alloc(length * length);
    noisePointer.alloc(length * length);
    double *pHost = pointer.host();
    double *pDev = pointer.dev();
    double *pcHost = covariancePointer.host();
    double *pcDev = covariancePointer.dev();
    double *pnHost = noisePointer.host();
    double *pnDev = noisePointer.dev();
    offset[0] = pHost;
    offset[count] = pDev;
    covarianceOffset[0] = pcHost;
    covarianceOffset[count] = pcDev;
    noiseOffset[0] = pnHost;
    noiseOffset[count] = pnDev;
    for (int i = 1; i < count; ++i)
    {
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
        offset[count + i] = offset[count + i - 1] + lengthPerOffset[i - 1];
        covarianceOffset[i] = covarianceOffset[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        covarianceOffset[count + i] = covarianceOffset[count + i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        noiseOffset[i] = noiseOffset[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        noiseOffset[count + i] = noiseOffset[count + i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
    }
    for (std::map<std::string, DataInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            for (int j = 0; j < lengthPerOffset[ii]; j++)
            {
                offset[ii][j] = (*i).second.data[j];
                covarianceOffset[ii][j * length + j] = (*i).second.covarianceData[j];
                noiseOffset[ii][j * length + j] = (*i).second.noiseData[j];
            }
        }
        else
        {
            std::cout << "Link failed in data of name: " + (*i).first + ".\n";
        }
    }
    pointer.copyHost2Dev(length);
    covariancePointer.copyHost2Dev(length * length);
    noisePointer.copyHost2Dev(length * length);
    instances.alloc((2 * length + 1) * length);
}

Pointer<double> Data::GetStatePointer()
{
    return pointer;
}

Pointer<double> Data::GetStateCovariancePointer()
{
    return covariancePointer;
}

Pointer<double> Data::GetStateCovariancePointer()
{
    return noisePointer;
}

Pointer<double> Data::GetInstances()
{
    return instances;
}

int Data::GetStateLength()
{
    return length;
}

int Data::GetSigmaLength()
{
    return 2 * length + 1;
}

void DataLoader::Add(std::string name, int length)
{
    info[name] = DataInfo(length);
}

void DataLoader::Link(std::string name, double *data, double *covarianceData, double *noiseData)
{
    info[name].data = data;
    info[name].covarianceData = covarianceData;
    info[name].noiseData = noiseData;
    info[name].linked = true;
}

void DataLoader::Remove(std::string name)
{
    info.erase(name);
}

Data DataLoader::Load()
{
    return Data(info);
}