#ifndef MATHGPU_HEADER
#define MATHGPU_HEADER

#include "../../structure/include/pointer.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define TOL8_GPU 1e-8
#define THREAD_COUNT 1024
#define CEIL(value, div) ((value + div - 1) / div)

namespace MathGPU
{
    static cublasHandle_t cublasHandle;
    static cusolverDnHandle_t cusolverDnHandle;

    void CreateHandles();
    void DestroyHandles();

    // Vector Zero
    void Zero(double *pv_o, int length);
    // Vector copy
    void Copy(double *pv_o, double *pv_i, int length);
    // Matrix Identity
    void Identity(double *m_o, int lengthI, int lengthJ);

    // Vector Element-wise Addition In-place
    void Add(double *pv_io, double *pv_i, int length);
    // Vector Element-wise Subtraction In-place
    void Sub(double *pv_io, double *pv_i, int length);
    // Vector Constant Multiplication In-place
    void Mul(double *pv_io, double v_i, int length);
    // Vector Element-wise Multiplication In-place
    void Mul(double *pv_io, double *pv_i, int length);

    // Vector Element-wise Addition Out-place
    void Add(double *pv_o, double *pvL_i, double *pvR_i, int length);
    // Vector Element-wise Subtraction Out-place
    void Sub(double *pv_o, double *pvL_i, double *pvR_i, int length);
    // Vector Constant Multiplication Out-place
    void Mul(double *pv_o, double *pvL_i, double vR_i, int length);
    // Vector Element-wise Multiplication Out-place
    void Mul(double *pv_o, double *pvL_i, double *pvR_i, int length);
    // Vector Constant Multiplication Addition Out-place
    void LRPO(double *pv_io, double *pvL_i, double vR_i, int length);
    // Vector Element-wise Multiplication Addition Out-place
    void LRPO(double *pv_io, double *pvL_i, double *pvR_i, int length);

    // Matrix Multiplication Natural (Line) x Natural (Column)
    void MatMulNN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Natural (Line) x Weight x Natural (Column)
    void MatMulNWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N);
    // Matrix Multiplication Natural (Line) x Transposed (Line)
    void MatMulNT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Natural (Line) x Weight x Transposed (Line)
    void MatMulNWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Natural (Column)
    void MatMulTN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Weight x Natural (Column)
    void MatMulTWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Transposed (Line)
    void MatMulTT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Weight x Transposed (Line)
    void MatMulTWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N);

    // Mean Operation
    void Mean(double *pv_o, double *pm_i, int lengthI, int lengthJ);
    // Mean with Weights Operation
    void Mean(double *pv_o, double *pm_i, double *pw_i, int lengthI, int lengthJ);
    // Dot Product Operation
    double Distance(double *pvL_i, double *pvR_i, int length);
    // Dot Operation
    double Dot(double *vL_i, double *vR_i, int length);
    // Comparison Operation
    bool Compare(double *pvL_i, double *pvR_i, int length);
    // Diagonalization Operation
    void Diag(double *pv_o, double *pm_i, int length);

    // Cholesky Decomposition
    void CholeskyDecomposition(double *pm_o, double *pm_i, int length);
    // Solve Linear System
    void CholeskySolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N);
}

#endif