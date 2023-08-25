#include "../include/mathGPU.hpp"

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

__global__ void CUDA_Mean(double *pv_o, double *pm_i, unsigned int lengthI, unsigned int lengthJ)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < lengthI)
    {
        double acc = 0.0;
        for (unsigned j = 0; j < lengthJ; ++j)
        {
            acc += pm_i[j * lengthI + index];
        }
        pv_o[index] = acc / lengthJ;
    }
}

__global__ void CUDA_Mean(double *pv_o, double *pm_i, double *pw_i, unsigned int lengthI, unsigned int lengthJ)
{
    unsigned index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < lengthI)
    {
        double acc = 0.0;
        for (unsigned j = 0; j < lengthJ; ++j)
        {
            acc += pw_i[j] * pm_i[j * lengthI + index];
        }
        pv_o[index] = acc;
    }
}

void MathGPU::CreateHandles()
{
    cublasCreate(&cublasHandle);
    cusolverDnCreate(&cusolverDnHandle);
}
void MathGPU::DestroyHandles()
{
    cublasDestroy(cublasHandle);
    cusolverDnDestroy(cusolverDnHandle);
}

void MathGPU::Zero(double *pv_o, int length)
{
    cudaMemsetAsync(pv_o, 0, sizeof(double) * length, cudaStreamDefault);
}
void MathGPU::Copy(double *pv_o, double *pv_i, int length)
{
    cudaMemcpyAsync(pv_o, pv_i, sizeof(double) * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice, cudaStreamDefault);
}

void MathGPU::Add(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, cudaStreamDefault>>>(pv_io, pv_i, length);
}
void MathGPU::Sub(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, cudaStreamDefault>>>(pv_io, pv_i, length);
}
void MathGPU::Mul(double *pv_io, double v_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, cudaStreamDefault>>>(pv_io, v_i, length);
}
void MathGPU::Mul(double *pv_io, double *pv_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, cudaStreamDefault>>>(pv_io, pv_i, length);
}
void MathGPU::Add(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Add<<<B, T, 0, cudaStreamDefault>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::Sub(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Sub<<<B, T, 0, cudaStreamDefault>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::Mul(double *pv_o, double *pvL_i, double vR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, cudaStreamDefault>>>(pv_o, pvL_i, vR_i, length);
}
void MathGPU::Mul(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    CUDA_Mul<<<B, T, 0, cudaStreamDefault>>>(pv_o, pvL_i, pvR_i, length);
}
void MathGPU::MatMulNN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, M, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulNWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *aux;
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cudaMallocAsync(&aux, sizeof(double) * min(M, N) * K, cudaStreamDefault);
    Zero(aux, min(M, N) * K);
    if (M < N)
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, M, K, pmL_i, M, pw_i, 1, aux, M);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, aux, M, pmR_i, K, &beta, pm_o, M);
    }
    else
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_LEFT, K, N, pmR_i, K, pw_i, 1, aux, K);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, M, aux, K, &beta, pm_o, M);
    }
    cudaFreeAsync(aux, cudaStreamDefault);
}
void MathGPU::MatMulNT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, M, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::MatMulNWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *aux;
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cudaMallocAsync(&aux, sizeof(double) * min(M, N) * K, cudaStreamDefault);
    Zero(aux, min(M, N) * K);
    if (M < N)
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, M, K, pmL_i, M, pw_i, 1, aux, M);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, aux, M, pmR_i, N, &beta, pm_o, M);
    }
    else
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, N, K, pmR_i, M, pw_i, 1, aux, M);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, M, aux, N, &beta, pm_o, M);
    }
    cudaFreeAsync(aux, cudaStreamDefault);
}
void MathGPU::MatMulTN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, K, pmR_i, K, &beta, pm_o, M);
}
void MathGPU::MatMulTWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *aux;
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cudaMallocAsync(&aux, sizeof(double) * min(M, N) * K, cudaStreamDefault);
    Zero(aux, min(M, N) * K);
    if (M < N)
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_LEFT, K, M, pmL_i, K, pw_i, 1, aux, K);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, aux, K, pmR_i, K, &beta, pm_o, M);
    }
    else
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_LEFT, K, N, pmR_i, K, pw_i, 1, aux, K);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, pmL_i, K, aux, K, &beta, pm_o, M);
    }
    cudaFreeAsync(aux, cudaStreamDefault);
}
void MathGPU::MatMulTT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, K, pmR_i, N, &beta, pm_o, M);
}
void MathGPU::MatMulTWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *aux;
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cudaMallocAsync(&aux, sizeof(double) * min(M, N) * K, cudaStreamDefault);
    Zero(aux, min(M, N) * K);
    if (M < N)
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_LEFT, K, M, pmL_i, K, pw_i, 1, aux, K);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N, M, N, K, &alpha, aux, K, pmR_i, N, &beta, pm_o, M);
    }
    else
    {
        cublasDdgmm(cublasHandle, cublasSideMode_t::CUBLAS_SIDE_RIGHT, N, K, pmR_i, M, pw_i, 1, aux, M);
        cublasDgemm(cublasHandle, cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_T, M, N, K, &alpha, pmL_i, K, aux, N, &beta, pm_o, M);
    }
    cudaFreeAsync(aux, cudaStreamDefault);
}
void MathGPU::Mean(double *pv_o, double *pm_i, int lengthI, int lengthJ)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(lengthI, T.x));
    CUDA_Mean<<<B, T, 0, cudaStreamDefault>>>(pv_o, pm_i, lengthI, lengthJ);
}
void MathGPU::Mean(double *pv_o, double *pm_i, double *pw_i, int lengthI, int lengthJ)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(lengthI, T.x));
    CUDA_Mean<<<B, T, 0, cudaStreamDefault>>>(pv_o, pm_i, pw_i, lengthI, lengthJ);
}
bool MathGPU::Compare(double *pvL_i, double *pvR_i, int length)
{
    dim3 T(THREAD_COUNT);
    dim3 B(CEIL(length, T.x));
    double *pDev;
    double res = -1.0;
    cublasSetStream(cublasHandle, cudaStreamDefault);
    cudaMallocAsync(&pDev, sizeof(double) * length, cudaStreamDefault);
    CUDA_Sub<<<B, T, 0, cudaStreamDefault>>>(pDev, pvL_i, pvR_i, length);
    cublasDnrm2(cublasHandle, length, pDev, 1, &res);
    cudaFreeAsync(pDev, cudaStreamDefault);
    return true;
}
void MathGPU::Diag(double *pv_o, double *pm_i, int length)
{
    cublasDcopy(cublasHandle, length, pm_i, length + 1, pv_o, 1);
}
void MathGPU::CholeskyDecomposition(double *pm_o, double *pm_i, int length)
{
    int size = 0;
    int *pdInfo;
    double *pdAux;
    cusolverDnSetStream(cusolverDnHandle, cudaStreamDefault);
    cudaMemcpyAsync(pm_o, pm_i, sizeof(double) * length * length, cudaMemcpyKind::cudaMemcpyDeviceToDevice, cudaStreamDefault);
    cusolverDnDpotrf_bufferSize(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, &size);
    cudaMallocAsync(&pdInfo, sizeof(int), cudaStreamDefault);
    cudaMallocAsync(&pdAux, sizeof(double) * size, cudaStreamDefault);
    cusolverDnDpotrf(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, length, pm_o, length, pdAux, size, pdInfo);
    cudaFreeAsync(pdAux, cudaStreamDefault);
    cudaFreeAsync(pdInfo, cudaStreamDefault);
}
void MathGPU::CholeskySolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    double *pm;
    int *pdInfo;
    cusolverDnSetStream(cusolverDnHandle, cudaStreamDefault);
    cudaMallocAsync(&pm, sizeof(double) * M * K, cudaStreamDefault);
    cudaMallocAsync(&pdInfo, sizeof(int), cudaStreamDefault);
    CholeskyDecomposition(pm, pmL_i, K);
    cudaMemcpyAsync(pm_o, pmR_i, sizeof(double) * M * N, cudaMemcpyKind::cudaMemcpyDeviceToDevice, cudaStreamDefault);
    cusolverDnDpotrs(cusolverDnHandle, cublasFillMode_t::CUBLAS_FILL_MODE_LOWER, M, N, pm, M, pm_o, M, pdInfo);
    cudaFreeAsync(pdInfo, cudaStreamDefault);
    cudaFreeAsync(pm, cudaStreamDefault);
    cudaDeviceSynchronize();
}
