#include "../include/data.hpp"

DataLoader::DataLoader() {}

Info::Info() : length(0), data(NULL), linked(false) {}

Info::Info(int length) : length(length), data(NULL), linked(false) {}

Data::Data(std::map<std::string, Info> &info) : count(info.size()), length(0)
{
    if (count == 0)
    {
        return;
    }
    offset = (double **)malloc(sizeof(double *) * 2 * count);
    covarianceOffset = (double **)malloc(sizeof(double *) * 2 * count);
    lengthPerOffset = (int *)malloc(sizeof(int) * count);
    int iaux = 0;
    for (std::map<std::string, Info>::iterator i = info.begin(); i != info.end(); ++i)
    {
        index[(*i).first] = iaux;
        int l = (*i).second.length;
        lengthPerOffset[iaux] = l;
        length += l;
        ++iaux;
    }
    pointer.alloc(length);
    covariancePointer.alloc(length * length);
    double *pHost = pointer.host();
    double *pDev = pointer.dev();
    double *pcHost = covariancePointer.host();
    double *pcDev = covariancePointer.dev();
    offset[0] = pHost;
    offset[count] = pDev;
    covarianceOffset[0] = pcHost;
    covarianceOffset[count] = pcDev;
    for (int i = 1; i < count; ++i)
    {
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
        offset[count + i] = offset[count + i - 1] + lengthPerOffset[i - 1];
        covarianceOffset[i] = covarianceOffset[i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
        covarianceOffset[count + i] = covarianceOffset[count + i - 1] + lengthPerOffset[i - 1] * length + lengthPerOffset[i - 1];
    }
    for (std::map<std::string, Info>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            for (int j = 0; j < lengthPerOffset[ii]; j++)
            {
                offset[ii][j] = (*i).second.data[j];
                covarianceOffset[ii][j * length + j] = (*i).second.covarianceData[j];
            }
        }
        else
        {
            std::cout << "Link failed in data of name: " + (*i).first + ".\n";
        }
    }
    pointer.copyHost2Dev(length);
    covariancePointer.copyHost2Dev(length * length);
    instances.alloc((2*length+1) * length);
}

void DataLoader::Add(std::string name, int length)
{
    info[name] = Info(length);
}

void DataLoader::Link(std::string name, double *data, double *covarianceData)
{
    info[name].data = data;
    info[name].covarianceData = covarianceData;
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