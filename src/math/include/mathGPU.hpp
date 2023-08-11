#ifndef MATHGPU_HEADER
#define MATHGPU_HEADER

#include "../../structure/include/pointer.hpp"
#include <cuda.h>
#include <cublas.h>
#include <cusolverDn.h>

namespace MathGPU
{
    // Vector Element-wise Addition In-place
    void Add(Pointer<double> v_io, Pointer<double> v_i, int length);
    // Vector Element-wise Subtraction In-place
    void Sub(Pointer<double> v_io, Pointer<double> v_i, int length);
    // Vector Constant Multiplication In-place
    void Mul(Pointer<double> v_io, double v_i, int length);
    // Vector Element-wise Multiplication In-place
    void Mul(Pointer<double> v_io, Pointer<double> v_i, int length);

    // Vector Element-wise Addition Out-place
    void Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length);
    // Vector Element-wise Subtraction Out-place
    void Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length);
    // Vector Constant Multiplication Out-place
    void Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length);
    // Vector Element-wise Multiplication Out-place
    void Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length);

    // Matrix Multiplication
    void MatMul(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N);

    // Mean Operation
    void Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ);
    // Comparison Operation
    bool Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length);
    // Diagonalization Operation
    bool Diag(Pointer<double> v_o, Pointer<double> m_i, int length);

    // Cholesky Decomposition
    void CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length);
    // Solve Linear System
    void CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N);
}

#endif