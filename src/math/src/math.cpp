#include "../include/math.hpp"

void Math::Zero(Pointer<double> v_o, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Zero(v_o.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Zero(v_o.dev(), length);
    }
}
void Math::Copy(Pointer<double> v_o, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Copy(v_o.host(), v_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Copy(v_o.dev(), v_i.dev(), length);
    }
}
void Math::Add(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Add(v_io.host(), v_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Add(v_io.dev(), v_i.dev(), length);
    }
}
void Math::Sub(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Sub(v_io.host(), v_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Sub(v_io.dev(), v_i.dev(), length);
    }
}
void Math::Mul(Pointer<double> v_io, double v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_io.host(), v_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_io.dev(), v_i, length);
    }
}
void Math::Mul(Pointer<double> v_io, Pointer<double> v_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_io.host(), v_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_io.dev(), v_i.dev(), length);
    }
}
void Math::Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Add(v_o.host(), vL_i.host(), vR_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Add(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
    }
}
void Math::Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Sub(v_o.host(), vL_i.host(), vR_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Sub(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
    }
}
void Math::Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_o.host(), vL_i.host(), vR_i, length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_o.dev(), vL_i.dev(), vR_i, length);
    }
}
void Math::Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mul(v_o.host(), vL_i.host(), vR_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mul(v_o.dev(), vL_i.dev(), vR_i.dev(), length);
    }
}
void Math::MatMulNN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulNN(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulNN(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
void Math::MatMulNWN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, Pointer<double> w_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulNWN(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), w_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulNWN(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), w_i.host(), M, K, N);
    }
}
void Math::MatMulNT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulNT(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulNT(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
void Math::MatMulNWT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, Pointer<double> w_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulNWT(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), w_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulNWT(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), w_i.dev(), M, K, N);
    }
}
void Math::MatMulTN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulTN(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulTN(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
void Math::MatMulTWN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, Pointer<double> w_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulTWN(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), w_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulTWN(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), w_i.dev(), M, K, N);
    }
}
void Math::MatMulTT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulTT(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulTT(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
void Math::MatMulTWT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, Pointer<double> w_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::MatMulTWT(beta, m_o.host(), alpha, mL_i.host(), mR_i.host(), w_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::MatMulTWT(beta, m_o.dev(), alpha, mL_i.dev(), mR_i.dev(), w_i.dev(), M, K, N);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_io, Pointer<double> v_i, int length, Type type), Pointer<double> m_io, Pointer<double> m_i, int length, int iteration, int stride_io, int stride_i, int offset_io, int offset_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_io + (i * stride_io + offset_io), m_i + (i * stride_i + offset_i), length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_io, double v_i, int length, Type type), Pointer<double> m_io, double *v_i, int length, int iteration, int stride_io, int offset_io, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_io + (i * stride_io + offset_io), *(v_i + i), length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int length, int iteration, int stride_o, int strideL_i, int strideR_i, int offset_o, int offsetL_i, int offsetR_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_o + (i * stride_o + offset_o), mL_i + (i * strideL_i + offsetL_i), mR_i + (i * strideR_i + offsetR_i), length, type);
    }
}
void Math::Iterate(void (*op_i)(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length, Type type), Pointer<double> m_o, Pointer<double> mL_i, double *vR_i, int length, int iteration, int stride_o, int strideL_i, int offset_o, int offsetL_i, Type type)
{
    for (int i = 0; i < iteration; ++i)
    {
        op_i(m_o + (i * stride_o + offset_o), mL_i + (i * strideL_i + offsetL_i), *(vR_i + i), length, type);
    }
}
void Math::Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mean(v_o.host(), m_i.host(), lengthI, lengthJ);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mean(v_o.dev(), m_i.dev(), lengthI, lengthJ);
    }
}
void Math::Mean(Pointer<double> v_o, Pointer<double> m_i, Pointer<double> w_i, int lengthI, int lengthJ, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Mean(v_o.host(), m_i.host(), w_i.host(), lengthI, lengthJ);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Mean(v_o.dev(), m_i.dev(), w_i.dev(), lengthI, lengthJ);
    }
}
bool Math::Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        return MathCPU::Compare(vL_i.host(), vR_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        return MathGPU::Compare(vL_i.dev(), vR_i.dev(), length);
    }
    return false;
}
void Math::Diag(Pointer<double> v_o, Pointer<double> m_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::Diag(v_o.host(), m_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::Diag(v_o.dev(), m_i.dev(), length);
    }
}
void Math::LUDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::LUDecomposition(m_o.host(), m_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::LUDecomposition(m_o.dev(), m_i.dev(), length);
    }
}
void Math::LUSolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::LUSolver(m_o.host(), mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::LUSolver(m_o.dev(), mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
void Math::CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::CholeskyDecomposition(m_o.host(), m_i.host(), length);
    }
    else if (type == Type::GPU)
    {
        MathGPU::CholeskyDecomposition(m_o.dev(), m_i.dev(), length);
    }
}
void Math::CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N, Type type)
{
    if (type == Type::CPU)
    {
        MathCPU::CholeskySolver(m_o.host(), mL_i.host(), mR_i.host(), M, K, N);
    }
    else if (type == Type::GPU)
    {
        MathGPU::CholeskySolver(m_o.dev(), mL_i.dev(), mR_i.dev(), M, K, N);
    }
}
