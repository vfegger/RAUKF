#include "../include/mathCPU.hpp"

void MathCPU::Copy(Pointer<double> v_o, Pointer<double> v_i, int length)
{
    double *pv_o = v_o.host();
    double *pv_i = v_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pv_i[i];
    }
}

void MathCPU::Add(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    double *pv_io = v_io.host();
    double *pv_i = v_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] += pv_i[i];
    }
}
void MathCPU::Sub(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    double *pv_io = v_io.host();
    double *pv_i = v_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] -= pv_i[i];
    }
}
void MathCPU::Mul(Pointer<double> v_io, double v_i, int length)
{
    double *pv_io = v_io.host();
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] *= v_i;
    }
}
void MathCPU::Mul(Pointer<double> v_io, Pointer<double> v_i, int length)
{
    double *pv_io = v_io.host();
    double *pv_i = v_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] *= pv_i[i];
    }
}
void MathCPU::Add(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    double *pv_o = v_o.host();
    double *pvL_i = vL_i.host();
    double *pvR_i = vR_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] + pvR_i[i];
    }
}
void MathCPU::Sub(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    double *pv_o = v_o.host();
    double *pvL_i = vL_i.host();
    double *pvR_i = vR_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] - pvR_i[i];
    }
}
void MathCPU::Mul(Pointer<double> v_o, Pointer<double> vL_i, double vR_i, int length)
{
    double *pv_o = v_o.host();
    double *pvL_i = vL_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] * vR_i;
    }
}
void MathCPU::Mul(Pointer<double> v_o, Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    double *pv_o = v_o.host();
    double *pvL_i = vL_i.host();
    double *pvR_i = vR_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] * pvR_i[i];
    }
}
void MathCPU::MatMulNN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    double *pm_o = m_o.host();
    double *pmL_i = mL_i.host();
    double *pmR_i = mR_i.host();
    double *auxL = (double *)malloc(sizeof(double) * M * K);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            auxL[i * K + j] = pmL_i[j * M + i];
        }
    }
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += auxL[i * K + k] * pmR_i[j * K + k];
            }
            pm_o[j * M + i] = beta * pm_o[j * M + i] + alpha * acc;
        }
    }
    free(auxL);
}
void MathCPU::MatMulNT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    double *pm_o = m_o.host();
    double *pmL_i = mL_i.host();
    double *pmR_i = mR_i.host();
    double *auxL = (double *)malloc(sizeof(double) * M * K);
    double *auxR = (double *)malloc(sizeof(double) * K * N);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            auxL[i * K + j] = pmL_i[j * M + i];
        }
    }
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            auxR[i * K + j] = pmR_i[j * N + i];
        }
    }
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += auxL[i * K + k] * pmR_i[j * K + k];
            }
            pm_o[j * M + i] = beta * pm_o[j * M + i] + alpha * acc;
        }
    }
}
void MathCPU::MatMulTN(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    double *pm_o = m_o.host();
    double *pmL_i = mL_i.host();
    double *pmR_i = mR_i.host();
    double acc;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += pmL_i[i * K + k] * pmR_i[j * K + k];
            }
            pm_o[j * M + i] = beta * pm_o[j * M + i] + alpha * acc;
        }
    }
}
void MathCPU::MatMulTT(double beta, Pointer<double> m_o, double alpha, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    double *pm_o = m_o.host();
    double *pmL_i = mL_i.host();
    double *pmR_i = mR_i.host();
    double *auxR = (double *)malloc(sizeof(double) * K * N);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            auxR[i * K + j] = pmR_i[j * N + i];
        }
    }
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += pmL_i[i * K + k] * auxR[j * K + k];
            }
            pm_o[j * M + i] = beta * pm_o[j * M + i] + alpha * acc;
        }
    }
}
void MathCPU::Mean(Pointer<double> v_o, Pointer<double> m_i, int lengthI, int lengthJ)
{
    double *pv_o = v_o.host();
    double *pm_i = m_i.host();
    for (int i = 0; i < lengthI; ++i)
    {
        pv_o[i] = 0.0;
    }
    for (int j = 0; j < lengthJ; ++j)
    {
        for (int i = 0; i < lengthI; ++i)
        {
            pv_o[i] += pm_i[j * lengthI + i];
        }
        for (int i = 0; i < lengthI; ++i)
        {
            pv_o[i] *= 1.0 / lengthJ;
        }
    }
}
bool MathCPU::Compare(Pointer<double> vL_i, Pointer<double> vR_i, int length)
{
    double *pvL_i = vL_i.host();
    double *pvR_i = vR_i.host();
    bool res = true;
    for (int i = 0; i < length; ++i)
    {
        res |= pvL_i[i] == pvR_i[i];
    }
}
bool MathCPU::Diag(Pointer<double> v_o, Pointer<double> m_i, int length)
{
    double *pv_o = v_o.host();
    double *pm_i = m_i.host();
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pm_i[i * length + i];
    }
}
void MathCPU::CholeskyDecomposition(Pointer<double> m_o, Pointer<double> m_i, int length)
{
    double *pm_o = m_o.host();
    double *pm_i = m_i.host();
    for (int j = 0; j < length; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < length; ++k)
        {
            sum += pm_o[k * length + j] * pm_o[k * length + j];
        }
        pm_o[j * length + j] = sqrt(pm_i[j * length + j] - sum);
        for (int i = j + 1; i < length; ++i)
        {
            sum = 0.0;
            for (int k = 0; k < j; ++k)
            {
                sum += pm_o[k * length + i] * pm_o[k * length + j];
            }
            pm_o[j * length + i] = (1.0 / pm_o[j * length + j]) * (pm_i[j * length + i] - sum);
        }
    }
}
void MathCPU::CholeskySolver(Pointer<double> m_o, Pointer<double> mL_i, Pointer<double> mR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    Pointer<double> m;
    m.alloc(M * K);
    CholeskyDecomposition(m, mL_i, K);

    double *pm = m.host();
    double *pm_o = m_o.host();
    double *pmR_i = mR_i.host();
    for (int k = 0; k < N; ++k)
    {
        for (int i = 0; i < M; ++i)
        {
            double sum = 0.0;
            for (int j = 0; j < i; ++j)
            {
                sum += pm[j * M + i] * pm_o[k * M + j];
            }
            pm_o[k * M + i] = (pmR_i[k * M + i] - sum) / pm[i * M + i];
        }
        for (int j = K - 1; j >= 0; --j)
        {
            double sum = 0.0;
            for (int i = j + 1; i < M; ++i)
            {
                sum += pm[j * M + i] * pm_o[k * M + i];
            }
            pm_o[k * M + j] = (pm_o[k * M + j] - sum) / pm[j * M + j];
        }
    }
    m.free();
}
