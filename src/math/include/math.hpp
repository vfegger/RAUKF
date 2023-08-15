#ifndef MATH_HEADER
#define MATH_HEADER

#include "../../structure/include/pointer.hpp"
#include "mathCPU.hpp"
#include "mathGPU.hpp"

namespace Math
{
    // Vector copy
    void Copy(Pointer<double> v_o, Pointer<double> v_i, int length, Type type);

    // Vector Element-wise Addition In-place
    void Add(Pointer<double> v_io, Pointer<double> v_i, int length, Type type);
    // Vector Element-wise Subtraction In-place
    void Sub(Pointer<double> v_io, Pointer<double> v_i, int length, Type type);
    // Vector Constant Multiplication In-place
    void Mul(Pointer<double> v_io, double v_i, int length, Type type);
    // Vector Element-wise Multiplication In-place
    void Mul(Pointer<double> v_io, Pointer<double> v_i, int length, Type type);

    // Vector Element-wise Addition Out-place
    void Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type);
    // Vector Element-wise Subtraction Out-place
    void Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type);
    // Vector Constant Multiplication Out-place
    void Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type);
    // Vector Element-wise Multiplication Out-place
    void Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type);

    // Matrix Multiplication Natural (Line) x Natural (Column)
    void MatMulNN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type);
    // Matrix Multiplication Natural (Line) x Transposed (Line)
    void MatMulNT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type);
    // Matrix Multiplication Transposed (Column) x Natural (Column)
    void MatMulTN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type);
    // Matrix Multiplication Transposed (Column) x Transposed (Line)
    void MatMulTT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type);

    // In-Place Operator Distribution
    void Iterate(void (*op_i)(Pointer<double> v_io, Pointer<double> v_i, int length, Type type), Pointer<double> m_io, Pointer<double> m_i, int length, int iteration, int stride_io, int stride_i, int offset_io, int offset_i, Type type);
    // In-Place Operator Distribution
    void Iterate(void (*op_i)(Pointer<double> v_io, double v_i, int length, Type type), Pointer<double> m_io, double *v_i, int length, int iteration, int stride_io, int offset_io, Type type);

    // Out-Place Operator Distribution
    void Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int length, int iteration, int stride_o, int strideL_i, int strideR_i, int offset_o, int offsetL_i, int offsetR_i, Type type);
    // Out-Place Operator Distribution
    void Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, double *vR_i, int length, int iteration, int stride_o, int strideL_i, int offset_o, int offsetL_i, Type type);

    // Mean Operation
    void Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ, Type type);
    // Comparison Operation
    bool Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type);
    // Diagonalization Operation
    bool Diag(Pointer<double> v_o, Pointer<double> m_i, int length, Type type);

    // Cholesky Decomposition
    void CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length, Type type);
    // Solve Linear System
    void CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type);
}

#endif