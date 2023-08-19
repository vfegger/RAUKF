#ifndef POINTER_HEADER
#define POINTER_HEADER

#include <stdlib.h>
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
        pHost = (T *)malloc(sizeof(T) * length);
        cudaMalloc(&pDev, sizeof(T) * length);
    }

    void free()
    {
        ::free(pHost);
        cudaFree(pDev);
    }

    void copyHost2Dev(unsigned length)
    {
        cudaMemcpy(pDev, pHost, sizeof(T) * length, cudaMemcpyKind::cudaMemcpyHostToDevice);
    }
    void copyDev2Host(unsigned length)
    {
        cudaMemcpy(pHost, pDev, sizeof(T) * length, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    }

    Pointer<T> operator+(int v_i)
    {
        return Pointer<T>(this->pHost + v_i, this->pDev + v_i);
    }
};

#endif