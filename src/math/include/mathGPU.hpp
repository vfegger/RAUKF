#ifndef MATHGPU_HEADER
#define MATHGPU_HEADER

#include "../../structure/include/pointer.hpp"
#include <cuda.h>
#include <cublas.h>
#include <cusolverDn.h>

#define TOL8_GPU 1e-8

namespace MathGPU
{
    // Vector copy
    void Copy(double *pv_o, double *pv_i, int length);

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

    // Matrix Multiplication Natural (Line) x Natural (Column)
    void MatMulNN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Natural (Line) x Transposed (Line)
    void MatMulNT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Natural (Column)
    void MatMulTN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);
    // Matrix Multiplication Transposed (Column) x Transposed (Line)
    void MatMulTT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N);

    // Mean Operation
    void Mean(double *pv_o, double *pm_i, int lengthI, int lengthJ);
    // Comparison Operation
    bool Compare(double *pvL_i, double *pvR_i, int length);
    // Diagonalization Operation
    bool Diag(double *pv_o, double *pm_i, int length);

    // Cholesky Decomposition
    void CholeskyDecomposition(double *pm_o, double *pm_i, int length);
    // Solve Linear System
    void CholeskySolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N);
}

#endif