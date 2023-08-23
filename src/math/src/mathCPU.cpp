#include "../include/mathCPU.hpp"

void MathCPU::Zero(double *pv_o, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = 0.0;
    }
}
void MathCPU::Copy(double *pv_o, double *pv_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pv_i[i];
    }
}

void MathCPU::Add(double *pv_io, double *pv_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] += pv_i[i];
    }
}
void MathCPU::Sub(double *pv_io, double *pv_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] -= pv_i[i];
    }
}
void MathCPU::Mul(double *pv_io, double v_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] *= v_i;
    }
}
void MathCPU::Mul(double *pv_io, double *pv_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_io[i] *= pv_i[i];
    }
}
void MathCPU::Add(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] + pvR_i[i];
    }
}
void MathCPU::Sub(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] - pvR_i[i];
    }
}
void MathCPU::Mul(double *pv_o, double *pvL_i, double vR_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] * vR_i;
    }
}
void MathCPU::Mul(double *pv_o, double *pvL_i, double *pvR_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pvL_i[i] * pvR_i[i];
    }
}
void MathCPU::MatMulNN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
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
void MathCPU::MatMulNWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *auxL = (double *)malloc(sizeof(double) * M * K);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            auxL[i * K + j] = pw_i[j] * pmL_i[j * M + i];
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
void MathCPU::MatMulNT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
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
    free(auxR);
    free(auxL);
}
void MathCPU::MatMulNWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *auxL = (double *)malloc(sizeof(double) * M * K);
    double *auxR = (double *)malloc(sizeof(double) * K * N);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            auxL[i * K + j] = pw_i[j] * pmL_i[j * M + i];
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
    free(auxR);
    free(auxL);
}
void MathCPU::MatMulTN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
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
void MathCPU::MatMulTWN(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double acc;
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            acc = 0.0;
            for (int k = 0; k < K; ++k)
            {
                acc += pw_i[k] * pmL_i[i * K + k] * pmR_i[j * K + k];
            }
            pm_o[j * M + i] = beta * pm_o[j * M + i] + alpha * acc;
        }
    }
}
void MathCPU::MatMulTT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, int M, int K, int N)
{
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
    free(auxR);
}
void MathCPU::MatMulTWT(double beta, double *pm_o, double alpha, double *pmL_i, double *pmR_i, double *pw_i, int M, int K, int N)
{
    double *auxR = (double *)malloc(sizeof(double) * K * N);
    double acc;
    for (int j = 0; j < K; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            auxR[i * K + j] = pw_i[j] * pmR_i[j * N + i];
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
    free(auxR);
}
void MathCPU::Mean(double *pv_o, double *pm_i, int lengthI, int lengthJ)
{
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
    }
    double w = 1.0 / lengthJ;
    for (int i = 0; i < lengthI; ++i)
    {
        pv_o[i] *= w;
    }
}
void MathCPU::Mean(double *pv_o, double *pm_i, double *pw_i, int lengthI, int lengthJ)
{
    for (int i = 0; i < lengthI; ++i)
    {
        long double acc = 0.0;
        for (int j = 0; j < lengthJ; ++j)
        {
            acc += pw_i[j] * pm_i[j * lengthI + i];
        }
        pv_o[i] = (double)acc;
    }
}
bool MathCPU::Compare(double *pvL_i, double *pvR_i, int length)
{
    bool res = true;
    for (int i = 0; i < length; ++i)
    {
        res |= pvL_i[i] == pvR_i[i];
    }
    return res;
}
void MathCPU::Diag(double *pv_o, double *pm_i, int length)
{
    for (int i = 0; i < length; ++i)
    {
        pv_o[i] = pm_i[i * length + i];
    }
}
void MathCPU::LUPDecomposition(double *pm_io, int length, int *pP)
{
    int i, j, k, imax;
    double maxA, aux, absA;

    for (i = 0; i < length; ++i)
        pP[i] = i; // Unit permutation matrix, P[N] initialized with N

    for (i = 0; i < length; i++)
    {
        maxA = 0.0;
        imax = i;

        for (k = i; k < length; ++k)
            if ((absA = fabs(pm_io[i * length + k])) > maxA)
            {
                maxA = absA;
                imax = k;
            }

        if (maxA < TOL8_CPU)
        {
            return; // failure, matrix is degenerate
        }
        if (imax != i)
        {
            j = pP[i];
            pP[i] = pP[imax];
            pP[imax] = j;
            for (k = 0; k < length; ++k)
            {
                aux = pm_io[k * length + i];
                pm_io[k * length + i] = pm_io[k * length + imax];
                pm_io[k * length + imax] = aux;
            }
        }

        for (j = i + 1; j < length; j++)
        {
            pm_io[j * length + i] /= pm_io[i * length + i];

            for (k = i + 1; k < length; k++)
            {
                pm_io[j * length + k] -= pm_io[j * length + i] * pm_io[i * length + k];
            }
        }
    }
}
void MathCPU::LUPSolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    double *pm = (double *)malloc(sizeof(double) * M * K);
    int *P = (int *)malloc(sizeof(double) * K);
    for (int i = 0; i < M * K; ++i)
    {
        pm[i] = pmL_i[i];
    }
    LUPDecomposition(pm, K, P);

    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < K; i++)
        {
            pm_o[j * K + i] = pmR_i[j * K + P[i]];
            for (int k = 0; k < i; k++)
            {
                pm_o[j * K + i] -= pm[i * K + k] * pm_o[j * K + k];
            }
        }

        for (int i = K - 1; i >= 0; i--)
        {
            for (int k = i + 1; k < K; k++)
            {
                pm_o[j * K + i] -= pm[i * K + k] * pm_o[j * K + k];
            }
            pm_o[j * K + i] /= pm[i * K + i];
        }
    }
    free(P);
    free(pm);
}
void MathCPU::CholeskyDecomposition(double *pm_o, double *pm_i, int length)
{
    for (int j = 0; j < length; ++j)
    {
        double sum = 0.0;
        for (int k = 0; k < j; ++k)
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
void MathCPU::CholeskySolver(double *pm_o, double *pmL_i, double *pmR_i, int M, int K, int N)
{
    if (M != K)
    {
        return;
    }
    double *pm = (double *)malloc(sizeof(double) * M * K);
    CholeskyDecomposition(pm, pmL_i, K);

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
    free(pm);
}
