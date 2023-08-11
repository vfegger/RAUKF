#include "../include/math.hpp"

void Math::Add(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Add(v_io, v_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Add(v_io, v_i, length);
    }
}
void Math::Sub(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Sub(v_io, v_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Sub(v_io, v_i, length);
    }
}
void Math::Mul(Pointer<double> v_io, double v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_io, v_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_io, v_i, length);
    }
}
void Math::Mul(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_io, v_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_io, v_i, length);
    }
}
void Math::Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Add(v_o, vL_i, vR_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Add(v_o, vL_i, vR_i, length);
    }
}
void Math::Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Sub(v_o, vL_i, vR_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Sub(v_o, vL_i, vR_i, length);
    }
}
void Math::Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_o, vL_i, vR_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_o, vL_i, vR_i, length);
    }
}
void Math::Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_o, vL_i, vR_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_o, vL_i, vR_i, length);
    }
}
void Math::MatMul(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMul(m_o, mL_i, mR_i, M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMul(m_o, mL_i, mR_i, M, K, N);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_io, Pointer<double> v_i, int length, Type type), Pointer<double> m_io, Pointer<double> m_i, int length, int iteration, int stride_io, int stride_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_io + i * stride_io, m_i + i * stride_i, length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_io, double v_i, int length, Type type), Pointer<double> m_io, double *v_i, int length, int iteration, int stride_io, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_io + i * stride_io, *(v_i + i), length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int length, int iteration, int stride_o, int strideL_i, int strideR_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_o + i * stride_o, mL_i + i * strideL_i, mR_i + i * strideR_i, length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, double *vR_i, int length, int iteration, int stride_o, int strideL_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_o + i * stride_o, mL_i + i * strideL_i, *(vR_i + i), length, type);
    }
}
void Math::Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mean(v_o, m_i, lengthI, lengthJ);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mean(v_o, m_i, lengthI, lengthJ);
    }
}
bool Math::Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        return MathCPU::Compare(vL_i, vR_i, length);
    }
    else if (type == Type::GPU)
    {
        return MathGPU::Compare(vL_i, vR_i, length);
    }
}
bool Math::Diag(Pointer<double> v_o, Pointer<double> m_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Diag(v_o, m_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Diag(v_o, m_i, length);
    }
}
void Math::CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::CholeskyDecomposition(m_o, m_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::CholeskyDecomposition(m_o, m_i, length);
    }
}
void Math::CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::CholeskySolver(m_o, mL_i, mR_i, M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::CholeskySolver(m_o, mL_i, mR_i, M, K, N);
    }
}
