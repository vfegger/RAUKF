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

void MathGPU::Zero(double *pv_o, int length)
{
    cudaMemset(pv_o, 0, sizeof(double) * length);
}
void MathGPU::Copy(double *pv_o, double *pv_i, int length)
{
    cudaMemcpy(pv_o, pv_i, sizeof(double) * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

void MathGPU::Add(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, 0>>>(pv_io, pv_i, length);
}
void MathGPU::Sub(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, 0>>>(pv_io, pv_i, length);
}
void MathGPU::Mul(double *pv_io, double v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(pv_io, v_i, length);
}
void MathGPU::Mul(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(pv_io, pv_i, length);
}
void MathGPU::Add(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, 0>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::Sub(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, 0>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::Mul(double *pv_o, double *pvL_i, double vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(pv_o, pvL_i, vR_i, length);
}
void MathGPU::Mul(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, 0>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::MatMulNN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, M, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulNWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, M, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulNT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, M, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::MatMulNWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, M, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::MatMulTN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, K, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulTWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, K, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulTT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, K, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::MatMulTWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, K, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::Mean(double *pv_o, double *pm_i, int lengthI, int lengthJ)
{
    dim3 T(THREAD_COUNT);
    dim3 B(lengthJ);
    CUDA_Mean<THREAD_COUNT><<<B, T, THREAD_COUNT * sizeof(double), 0>>>(pv_o, pm_i, lengthI, lengthJ);
}
void MathGPU::Mean(double *pv_o, double *pm_i, double *pw_i, int lengthI, int lengthJ)
{
    dim3 T(THREAD_COUNT);
    dim3 B(lengthJ);
    CUDA_Mean<THREAD_COUNT><<<B, T, THREAD_COUNT * sizeof(double), 0>>>(pv_o, pm_i, lengthI, lengthJ);
}
bool MathGPU::Compare(double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    double *pDev;
    double res = -1.0;
    cudaMalloc(&pDev, sizeof(double) * length);
    CUDA_Sub<<<B, T, 0, 0>>>(pDev, pvL_i, pvR_i, length);
    cublasDnrm2(cublasHandle, length, pDev, 1, &res);
    cudaFree(pDev);
    return true;
}
void MathGPU::Diag(double *pv_o, double *pm_i, int length)
{
    cublasDcopy(cublasHandle, length, pm_i, length + 1, pv_o, 1);
}
void MathGPU::LUPDecomposition(double *pm_io, int length, int *pP)
{
}
void MathGPU::LUPSolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N)
{
}
void MathGPU::CholeskyDecomposition(double *pm_o, double *pm_i, int length)
{
    int size = 0;
    int *pdInfo;
    double *pdAux;
    cudaMemcpy(pm_o, pm_i, sizeof(double) * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    cusolverDnDpotrf_bufferSize(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, &size);
    cudaMalloc(&pdInfo, sizeof(int));
    cudaMalloc(&pdAux, sizeof(double) * size);
    cusolverDnDpotrf(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, pdAux, size, pdInfo);
    cudaFree(pdAux);
    cudaFree(pdInfo);
}
void MathGPU::CholeskySolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    double *pm;
    cudaMalloc(&pm, sizeof(double) * M * K);
    int *pdInfo;
    cudaMalloc(&pdInfo, sizeof(int));
    CholeskyDecomposition(pm, pmL_i, K);
    cudaMemcpy(pm_o, pmR_i, sizeof(double) * M * N, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
    cusolverDnDpotrs(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, M, N, pm, M, pm_o, M, pdInfo);
    cudaFree(pdInfo);
    cudaFree(pm);
}
