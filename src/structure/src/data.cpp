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
    offset = (int *)malloc(sizeof(int) * count);
    offset2 = (int *)malloc(sizeof(int) * count);
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
    offset[0] = 0;
    offset2[0] = 0;
    for (int i = 1; i < count; ++i)
    {
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
        offset2[i] = offset2[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
    }
    pointer.alloc(length);
    covariancePointer.alloc(length * length);
    noisePointer.alloc(length * length);
    instances.alloc((2 * length + 1) * length);
    double *pHost = pointer.host();
    double *pcHost = covariancePointer.host();
    double *pnHost = noisePointer.host();
    double *piHost = instances.host();
    for (int j = 0; j < length; ++j)
    {
        pHost[j] = 0.0;
        for (int i = 0; i < length; ++i)
        {
            pcHost[j * length + i] = 0.0;
            pnHost[j * length + i] = 0.0;
        }
    }
    for (int j = 0; j < 2 * length + 1; ++j)
    {
        for (int i = 0; i < length; ++i)
        {
            piHost[j * length + i] = 0.0;
        }
    }
    for (std::map<std::string, DataInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            double *aux0 = pHost + offset[ii];
            double *aux1 = pcHost + offset2[ii];
            double *aux2 = pnHost + offset2[ii];
            for (int j = 0; j < lengthPerOffset[ii]; j++)
            {
                aux0[j] = (*i).second.data[j];
                aux1[j * length + j] = (*i).second.covarianceData[j];
                aux2[j * length + j] = (*i).second.noiseData[j];
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
    cudaDeviceSynchronize();
}

Pointer<double> Data::GetStatePointer()
{
    return pointer;
}

Pointer<double> Data::GetStateCovariancePointer()
{
    return covariancePointer;
}

Pointer<double> Data::GetStateNoisePointer()
{
    return noisePointer;
}

void Data::GetStateData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = (pointer.host() + offset[ii]);
    int l = lengthPerOffset[ii];
    for (int i = 0; i < l; ++i)
    {
        data[i] = pHost[i];
    }
}

void Data::GetStateCovarianceData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = (covariancePointer.host() + offset2[ii]);
    int l = lengthPerOffset[ii];
    for (int i = 0; i < l; ++i)
    {
        data[i] = pHost[i * length + i];
    }
}

Pointer<double> Data::GetInstances()
{
    return instances;
}

void Data::SetInstances()
{
    instances.alloc((2 * length + 1) * length);
}

void Data::UnsetInstances()
{
    instances.free();
}

int Data::GetStateLength()
{
    return length;
}

int Data::GetSigmaLength()
{
    return 2 * length + 1;
}

int Data::GetOffset(std::string name)
{
    return offset[index[name]];
}

int Data::GetOffset2(std::string name)
{
    return offset2[index[name]];
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

Data *DataLoader::Load()
{
    return new Data(info);
}