#include "../include/mathGPU.hpp"

#define THREAD_COUNT 1024
#define CEIL(value, div) ((value + div - 1) / div)

__global__ void CUDA_Add(double *pv_io, double *pv_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_io[index] += pv_i[index];
    }
}
__global__ void CUDA_Sub(double *pv_io, double *pv_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_io[index] -= pv_i[index];
    }
}
__global__ void CUDA_Mul(double *pv_io, double v_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_io[index] *= v_i;
    }
}
__global__ void CUDA_Mul(double *pv_io, double *pv_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_io[index] *= pv_i[index];
    }
}

__global__ void CUDA_Add(double *pv_o, double *pvL_i, double *pvR_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_o[index] = pvL_i[index] + pvR_i[index];
    }
}
__global__ void CUDA_Sub(double *pv_o, double *pvL_i, double *pvR_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_o[index] = pvL_i[index] - pvR_i[index];
    }
}
__global__ void CUDA_Mul(double *pv_o, double *pvL_i, double vR_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_o[index] = pvL_i[index] * vR_i;
    }
}
__global__ void CUDA_Mul(double *pv_o, double *pvL_i, double *pvR_i, unsigned int length)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < length)
    {
        pv_o[index] = pvL_i[index] * pvR_i[index];
    }
}

template <unsigned int blockSize>
__device__ void warpReduce(volatile double *psv, unsigned int tid)
{
    if (blockSize >= 64u)
    {
        psv[tid] += psv[tid + 32u];
    }
    if (blockSize >= 32u)
    {
        psv[tid] += psv[tid + 16u];
    }
    if (blockSize >= 16u)
    {
        psv[tid] += psv[tid + 8u];
    }
    if (blockSize >= 8u)
    {
        psv[tid] += psv[tid + 4u];
    }
    if (blockSize >= 4u)
    {
        psv[tid] += psv[tid + 2u];
    }
    if (blockSize >= 2u)
    {
        psv[tid] += psv[tid + 1u];
    }
}

template <unsigned int blockSize>
__global__ void CUDA_Mean(double *pv_o, double *pm_i, unsigned int lengthI, unsigned int lengthJ)
{
    extern __shared__ double psv[];
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int stride = 2u * blockSize;
    unsigned int j;
    j = 2u * blockSize + tid;
    psv[tid] = 0;
    while (j < lengthJ)
    {
        psv[tid] += pm_i[(j)*lengthI + bid] + pm_i[(j + blockSize) * lengthI + bid];
        j += stride;
    }
    __syncthreads();
    if (blockSize >= 512u)
    {
        if (tid < 256u)
        {
            psv[tid] += psv[tid + 256u];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            psv[tid] += psv[tid + 128u];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            psv[tid] += psv[tid + 64u];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        warpReduce<blockSize>(psv, tid);
    }
    if (tid == 0)
    {
        pv_o[bid] = psv[0] / lengthJ;
    }
}

void MathGPU::Copy(Pointer<double> v_o, Pointer<double> v_i, int length)
{
    cudaMemcpy(v_o.dev(), v_i.dev(), sizeof(double) * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void MathGPU::Add(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, 0>>>(v_io.dev(), v_i.dev(), length);
}
void MathGPU::Sub(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, 0>>>(v_io.dev(), v_i.dev(), length);
}
void MathGPU::Mul(Pointer<double> v_io, double v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(v_io.dev(), v_i, length);
}
void MathGPU::Mul(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(v_io.dev(), v_i.dev(), length);
}
void MathGPU::Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, 0>>>(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
}
void MathGPU::Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, 0>>>(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
}
void MathGPU::Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(v_o.dev(), vL_i.dev(), vR_i, length);
}
void MathGPU::Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
}
void MathGPU::MatMulNN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    cublasHandle_t handle;
    cublasDgemm(handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, mL_i.dev(), M, mR_i.dev(), K, &beta, m_o.dev(), M);
}
void MathGPU::MatMulNT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    cublasHandle_t handle;
    cublasDgemm(handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, mL_i.dev(), M, mR_i.dev(), N, &beta, m_o.dev(), M);
}
void MathGPU::MatMulTN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    cublasHandle_t handle;
    cublasDgemm(handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, mL_i.dev(), K, mR_i.dev(), K, &beta, m_o.dev(), M);
}
void MathGPU::MatMulTT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    cublasHandle_t handle;
    cublasDgemm(handle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, mL_i.dev(), K, mR_i.dev(), N, &beta, m_o.dev(), M);
}
void MathGPU::Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ)
{
    dim3 T(THREAD_COUNT);
    dim3 B(lengthJ);
    CUDA_Mean<THREAD_COUNT><<<B, T, THREAD_COUNT * sizeof(double), 0>>>(v_o.dev(), m_i.dev(), lengthI, lengthJ);
}
bool MathGPU::Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    cublasHandle_t handle;
    double alpha = 1.0;
    double beta = 0.0;
    double *pDev;
    double res = -1.0;
    cudaMalloc(&pDev, sizeof(double) * length);
    CUDA_Sub<<<B, T, 0, 0>>>(pDev, vL_i.dev(), vR_i.dev(), length);
    cublasDnrm2(handle, length, pDev, 1, &res);
    cudaFree(pDev);
}
bool MathGPU::Diag(Pointer<double> v_o, Pointer<double> m_i, int length)
{
    cublasHandle_t handle;
    cublasDcopy(handle, length, m_i.dev(), length + 1, v_o.dev(), 1);
}
void MathGPU::CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length)
{
    cusolverDnHandle_t handle;
    int size = 0;
    int *pdInfo;
    double *pdAux;
    double *pm_o = m_o.dev();
    double *pm_i = m_i.dev();
    cudaMemcpy(pm_o, pm_i, sizeof(double) * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    cusolverDnDpotrf_bufferSize(handle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, &size);
    cudaMalloc(&pdInfo, sizeof(int));
    cudaMalloc(&pdAux, sizeof(double) * size);
    cusolverDnDpotrf(handle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, pdAux, size, pdInfo);
    cudaFree(pdAux);
    cudaFree(pdInfo);
}
void MathGPU::CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    cusolverDnHandle_t handle;
    Pointer<double> m;
    int *pdInfo;
    double *pm = m.dev();
    double *pm_o = m_o.dev();
    double *pmL_i = mL_i.dev();
    double *pmR_i = mR_i.dev();
    m.alloc(M * K);
    cudaMalloc(&pdInfo, sizeof(int));
    CholeskyDecomposition(m, mL_i, K);
    cudaMemcpy(pm_o, pmR_i, sizeof(double) * M * N, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    cusolverDnDpotrs(handle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, M, N, pm, M, pm_o, M, pdInfo);
    cudaFree(pdInfo);
    m.free();
}
