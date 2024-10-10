#ifndef POINTER_HEADER
#define POINTER_HEADER

#include <stdlib.h>
#include <iostream>
#include <cuda_runtime.h>

enum Type
{
    CPU,
    GPU
};

template <typename T>
struct Pointer
{
private:
    T *pHost;
    T *pDev;

    Pointer(T *pHost_i, T *pDev_i)
    {
        pHost = pHost_i;
        pDev = pDev_i;
    }

protected:
public:
    Pointer()
    {
        pHost = NULL;
        pDev = NULL;
    }

    T *host()
    {
        return pHost;
    }
    T *dev()
    {
        return pDev;
    }

    void alloc(unsigned length)
    {
        cudaError_t errh = cudaMallocHost(&pHost, sizeof(T) * length);
        cudaError_t errd = cudaMallocAsync(&pDev, sizeof(T) * length, cudaStreamDefault);
        if (errh != cudaSuccess || errd != cudaSuccess)
        {
            printf("Host error status is %s\n", cudaGetErrorString(errh));
            printf("Device error status is %s\n", cudaGetErrorString(errd));
            std::cout << "Allocation Failed.\n Host pointer: " << pHost << "Device Pointer:" << pDev << "\n";
            throw std::bad_alloc();
        }
    }

    void free()
    {
        cudaError_t errh = cudaFreeHost(pHost);
        cudaError_t errd = cudaFreeAsync(pDev, cudaStreamDefault);
        if (errh != cudaSuccess || errd != cudaSuccess)
        {
            printf("Host error status is %s\n", cudaGetErrorString(errh));
            printf("Device error status is %s\n", cudaGetErrorString(errd));
            std::cout << "Allocation Failed.\n Host pointer: " << pHost << "Device Pointer:" << pDev << "\n";
            throw std::bad_alloc();
        }
        pHost = NULL;
        pDev = NULL;
    }

    void copyHost2Dev(unsigned length)
    {
        cudaMemcpyAsync(pDev, pHost, sizeof(T) * length, cudaMemcpyKind::cudaMemcpyHostToDevice, cudaStreamDefault);
    }
    void copyDev2Host(unsigned length)
    {
        cudaMemcpyAsync(pHost, pDev, sizeof(T) * length, cudaMemcpyKind::cudaMemcpyDeviceToHost, cudaStreamDefault);
    }

    Pointer<T> operator+(int v_i)
    {
        return Pointer<T>(this->pHost + v_i, this->pDev + v_i);
    }
};

#endif