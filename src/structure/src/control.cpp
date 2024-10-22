#include "../include/control.hpp"

ControlLoader::ControlLoader() {}

ControlInfo::ControlInfo() : length(0), data(NULL), linked(false) {}

ControlInfo::ControlInfo(int length) : length(length), data(NULL), linked(false) {}

Control::Control(std::map<std::string, ControlInfo> &info) : count(info.size()), length(0)
{
    if (count == 0)
    {
        return;
    }
    offset = (int *)malloc(sizeof(int) * count);
    lengthPerOffset = (int *)malloc(sizeof(int) * count);
    int iaux = 0;
    for (std::map<std::string, ControlInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        index[(*i).first] = iaux;
        int l = (*i).second.length;
        lengthPerOffset[iaux] = l;
        length += l;
        ++iaux;
    }
    offset[0] = 0;
    for (int i = 1; i < count; ++i)
    {
        offset[i] = offset[i - 1] + lengthPerOffset[i - 1];
    }
    pointer.alloc(length);
    double *pHost = pointer.host();
    for (int j = 0; j < length; ++j)
    {
        pHost[j] = 0.0;
    }
    for (std::map<std::string, ControlInfo>::iterator i = info.begin(); i != info.end(); ++i)
    {
        if ((*i).second.linked)
        {
            int ii = index[(*i).first];
            double *aux0 = pHost + offset[ii];
            for (int j = 0; j < lengthPerOffset[ii]; j++)
            {
                aux0[j] = (*i).second.data[j];
            }
        }
        else
        {
            std::cout << "Link failed in data of name: " + (*i).first + ".\n";
        }
    }
    pointer.copyHost2Dev(length);
    cudaDeviceSynchronize();
}

Pointer<double> Control::GetControlPointer()
{
    return pointer;
}

void Control::SetControlData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = (pointer.host()+ offset[ii]);
    int l = lengthPerOffset[ii];
    for (int i = 0; i < l; ++i)
    {
        pHost[i] = data[i];
    }
}

void Control::GetControlData(std::string name, double *data)
{
    int ii = index[name];
    double *pHost = (pointer.host() + offset[ii]);
    int l = lengthPerOffset[ii];
    for (int i = 0; i < l; ++i)
    {
        data[i] = pHost[i];
    }
}

Pointer<double> Control::GetControlData()
{
    pointer.copyHost2Dev(length);
    cudaDeviceSynchronize();
    return pointer;
}

int Control::GetControlLength()
{
    return length;
}

int Control::GetOffset(std::string name)
{
    return offset[index[name]];
}

void ControlLoader::Add(std::string name, int length)
{
    info[name] = ControlInfo(length);
}

void ControlLoader::Link(std::string name, double *data)
{
    info[name].data = data;
    info[name].linked = true;
}

void ControlLoader::Remove(std::string name)
{
    info.erase(name);
}

Control *ControlLoader::Load()
{
    return new Control(info);
}