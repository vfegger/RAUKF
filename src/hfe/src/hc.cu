#include "../include/hc.hpp"

__host__ __device__ inline double C(double T_in)
{
    return 1324.75 * T_in + 3557900.0;
}

__host__ __device__ inline double K(double T_in)
{
    return (14e-3 + 2.517e-6 * T_in) * T_in + 12.45;
}

__host__ __device__ inline double DiffK(double TN_in, double T_in, double TP_in, double delta_in)
{
    double auxN = 2.0 * (K(TN_in) * K(T_in)) / (K(TN_in) + K(T_in)) * (TN_in - T_in) / delta_in;
    double auxP = 2.0 * (K(TP_in) * K(T_in)) / (K(TP_in) + K(T_in)) * (TP_in - T_in) / delta_in;
    return (auxN + auxP) / delta_in;
}

void HC::CPU::Diff(double *dT, double *dQ, double *T, double *Q, HCParms &parms)
{
    int index, offset;
    double T0, TiN, TiP, TjN, TjP, TkN, TkP;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lz = parms.Lz;
    int Lxy = Lx * Ly;
    double dx = parms.dx;
    double dy = parms.dy;
    double dz = parms.dz;
    double dt = parms.dt;
    double amp = parms.amp;
    // Difusion Contribution
    for (int k = 0; k < Lz; ++k)
    {
        for (int j = 0; j < Ly; ++j)
        {
            for (int i = 0; i < Lx; ++i)
            {
                index = (k * Ly + j) * Lx + i;
                T0 = T[index];
                TiN = (i != 0) ? T[index - 1] : T0;
                TiP = (i != Lx - 1) ? T[index + 1] : T0;
                TjN = (j != 0) ? T[index - Lx] : T0;
                TjP = (j != Ly - 1) ? T[index + Lx] : T0;
                TkN = (k != 0) ? T[index - Lxy] : T0;
                TkP = (k != Lz - 1) ? T[index + Lxy] : T0;

                dT[index] = DiffK(TiN, T0, TiP, dx) + DiffK(TjN, T0, TjP, dy) + DiffK(TkN, T0, TkP, dz);
            }
        }
    }
    // Heat Flux Contribution
    offset = (Lz - 1) * Lxy;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            index = j * Lx + i;
            dT[offset + index] += (amp / dz) * Q[index];
        }
    }
    // Retrieves the temporal derivative of the temperature
    for (int k = 0; k < Lz; ++k)
    {
        for (int j = 0; j < Ly; ++j)
        {
            for (int i = 0; i < Lx; ++i)
            {
                index = (k * Ly + j) * Lx + i;
                dT[index] /= C(T[index]);
            }
        }
    }
    // Retrieves the temporal derivative of the heat flux
    offset = Lz * Lxy;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            index = j * Lx + i;
            dQ[index] = distribution(generator) / dt;
        }
    }
}

void HC::CPU::AllocWorkspaceEuler(double *&workspace, HCParms &parms)
{
    workspace = (double *)malloc(sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1));
}

void HC::CPU::FreeWorkspaceEuler(double *workspace)
{
    free(workspace);
}

void HC::CPU::Euler(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    double dt = parms.dt;
    Diff(workspace, workspace + Lxyz, T, Q, parms);
    double *work = workspace;
    for (int i = 0; i < Lxyz; ++i)
    {
        T[i] = T[i] + dt * work[i];
    }
    work = workspace + Lxyz;
    for (int i = 0; i < Lxy; ++i)
    {
        Q[i] = Q[i] + dt * work[i];
    }
}

void HC::CPU::AllocWorkspaceRK4(double *&workspace, HCParms &parms)
{
    workspace = (double *)malloc(5 * sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1));
}

void HC::CPU::FreeWorkspaceRK4(double *workspace)
{
    free(workspace);
}

void HC::CPU::RK4(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    int L = Lxyz + Lxy;
    double dt = parms.dt;
    double *K1, *K2, *K3, *K4, *auxT, *auxQ, *work;
    K1 = workspace;
    K2 = K1 + L;
    K3 = K2 + L;
    K4 = K3 + L;
    auxT = K4 + L;
    auxQ = auxT + Lxyz;

    Diff(K1, K1 + Lxyz, T, Q, parms);
    work = K1;
    for (int i = 0; i < Lxyz; ++i)
    {
        auxT[i] = T[i] + 0.5 * dt * work[i];
    }
    work = K1 + Lxyz;
    for (int i = 0; i < Lxy; ++i)
    {
        auxQ[i] = Q[i] + 0.5 * dt * work[i];
    }
    Diff(K2, K2 + Lxyz, auxT, auxQ, parms);

    work = K2;
    for (int i = 0; i < Lxyz; ++i)
    {
        auxT[i] = T[i] + 0.5 * dt * work[i];
    }
    work = K2 + Lxyz;
    for (int i = 0; i < Lxy; ++i)
    {
        auxQ[i] = Q[i] + 0.5 * dt * work[i];
    }
    Diff(K3, K3 + Lxyz, auxT, auxQ, parms);

    work = K3;
    for (int i = 0; i < Lxyz; ++i)
    {
        auxT[i] = T[i] + dt * work[i];
    }
    work = K3 + Lxyz;
    for (int i = 0; i < Lxy; ++i)
    {
        auxQ[i] = Q[i] + dt * work[i];
    }
    Diff(K4, K4 + Lxyz, auxT, auxQ, parms);

    for (int i = 0; i < Lxyz; ++i)
    {
        T[i] = T[i] + (dt / 6) * (K1[i] + 2.0 * (K2[i] + K3[i]) + K4[i]);
    }
    K1 = K1 + Lxyz;
    K2 = K2 + Lxyz;
    K3 = K3 + Lxyz;
    K4 = K4 + Lxyz;
    for (int i = 0; i < Lxy; ++i)
    {
        Q[i] = Q[i] + (dt / 6) * (K1[i] + 2.0 * (K2[i] + K3[i]) + K4[i]);
    }
}

void HC::CPU::AllocWorkspaceRKF45(double *&workspace, HCParms &parms)
{
    workspace = (double *)malloc(8 * sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1));
}

void HC::CPU::FreeWorkspaceRKF45(double *workspace)
{
    free(workspace);
}

inline int bI(int i)
{
    return (i * i - i) / 2;
}

void RKTable(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref1, double *ref2, int L1, int L2, HC::HCParms &parms)
{
    int L = L1 + L2;
    for (int d = 0; d < n; ++d)
    {
        for (int i = 0; i < L1; ++i)
        {
            aux00[i] = ref1[i];
        }
        for (int i = 0; i < L2; ++i)
        {
            aux01[i] = ref2[i];
        }
        double *K_d = K + L * d;
        for (int j = 0; j < d; ++j)
        {
            double *K_j = K + L * j;
            for (int i = 0; i < L1; ++i)
            {
                aux00[i] += B[bI(d) + j] * h * K_j[i];
            }
            for (int i = 0; i < L2; ++i)
            {
                aux01[i] += B[bI(d) + j] * h * K_j[L1 + i];
            }
        }
        HC::CPU::Diff(K_d, K_d + L1, aux00, aux01, parms);
    }
    for (int i = 0; i < L1; ++i)
    {
        aux10[i] = aux00[i] = ref1[i];
    }
    for (int i = 0; i < L2; ++i)
    {
        aux11[i] = aux01[i] = ref2[i];
    }
    for (int j = 0; j < n; ++j)
    {
        double *K_j = K + L * j;
        for (int i = 0; i < L1; ++i)
        {
            aux00[i] += C[j] * h * K_j[i];
            aux10[i] += C[n + j] * h * K_j[i];
        }
        for (int i = 0; i < L2; ++i)
        {
            aux01[i] += C[j] * h * K_j[L1 + i];
            aux11[i] += C[n + j] * h * K_j[L1 + i];
        }
    }
}

void HC::CPU::RKF45(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    int L = Lxyz + Lxy;
    double dt = parms.dt;
    double *K, *auxT, *auxQ, *auxT_5, *auxQ_5;
    K = workspace;
    auxT = K + 6 * L;
    auxQ = auxT + Lxyz;
    auxT_5 = auxQ + Lxy;
    auxQ_5 = auxT + Lxyz;

    double h, hacc;
    h = dt;
    hacc = 0.0;
    double B[] = {0.25, 3.0 / 32.0, 9.0 / 32.0, 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0};
    double C[] = {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4101.0, -1.0 / 5.0, 0.0, 16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
    do
    {
        parms.dt = h;
        RKTable(6, B, C, h, K, auxT, auxQ, auxT_5, auxQ_5, T, Q, Lxyz, Lxy, parms);

        double acc = 0.0;
        for (int i = 0; i < Lxyz; ++i)
        {
            double temp = (auxT_5[i] - auxT[i]);
            acc += temp * temp;
        }
        for (int i = 0; i < Lxy; ++i)
        {
            double temp = (auxQ_5[i] - auxQ[i]);
            acc += temp * temp;
        }
        double e = TOL8_CPU / (2.0 * sqrt(acc) + 1e-3 * TOL8_CPU);
        double alpha = sqrt(sqrt(e));
        if (e > 1.0 || h < 1e-3 * dt)
        {
            hacc += h;
            h = dt - hacc;
            for (int i = 0; i < Lxyz; ++i)
            {
                T[i] = auxT[i];
            }
            for (int i = 0; i < Lxy; ++i)
            {
                Q[i] = auxQ[i];
            }
        }
        else
        {
            h = e * h;
        }
    } while (dt - hacc > 0);
    parms.dt = dt;
}