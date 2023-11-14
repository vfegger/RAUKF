#include "../include/hc.hpp"

__host__ __device__ inline double HC::C(double T_i)
{
    return 1324.75 * T_i + 3557900.0;
}
__host__ __device__ inline double HC::K(double T_i)
{
    return (14e-3 + 2.517e-6 * T_i) * T_i + 12.45;
}
__host__ __device__ inline double DiffK(double TN_i, double T_i, double TP_i, double delta_i)
{
    double auxN = 2.0 * (HC::K(TN_i) * HC::K(T_i)) / (HC::K(TN_i) + HC::K(T_i)) * (TN_i - T_i) / delta_i;
    double auxP = 2.0 * (HC::K(TP_i) * HC::K(T_i)) / (HC::K(TP_i) + HC::K(T_i)) * (TP_i - T_i) / delta_i;
    return (auxN + auxP) / delta_i;
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
            dQ[index] = 0.0;
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
void HC::CPU::ButcherTableau(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref0, double *ref1, int L0, int L1, HCParms &parms)
{
    int L = L0 + L1;
    for (int d = 0; d < n; ++d)
    {
        double *K_d = K + L * d;
        MathCPU::Copy(aux00, ref0, L0);
        MathCPU::Copy(aux01, ref1, L1);
        for (int j = 0; j < d; ++j)
        {
            double *K_j = K + L * j;
            MathCPU::LRPO(aux00, K_j, B[bI(d) + j] * h, L0);
            MathCPU::LRPO(aux01, K_j + L0, B[bI(d) + j] * h, L1);
        }
        HC::CPU::Diff(K_d, K_d + L0, aux00, aux01, parms);
    }
    MathCPU::Copy(aux00, ref0, L0);
    MathCPU::Copy(aux10, ref0, L0);
    MathCPU::Copy(aux01, ref1, L1);
    MathCPU::Copy(aux11, ref1, L1);
    for (int j = 0; j < n; ++j)
    {
        double *K_j = K + L * j;
        double temp = C[j] * h;
        MathCPU::LRPO(aux00, K_j, C[j] * h, L0);
        MathCPU::LRPO(aux01, K_j + L0, C[j] * h, L1);
        MathCPU::LRPO(aux10, K_j, C[n + j] * h, L0);
        MathCPU::LRPO(aux11, K_j + L0, C[n + j] * h, L1);
    }
}
void HC::CPU::RKF45(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    int L = Lxyz + Lxy;
    double dt = parms.dt;
    double *K, *aux, *auxT, *auxQ, *aux_5, *auxT_5, *auxQ_5;
    K = workspace;
    aux = K + 6 * L;
    auxT = aux;
    auxQ = auxT + Lxyz;
    aux_5 = aux + L;
    auxT_5 = auxQ + Lxy;
    auxQ_5 = auxT_5 + Lxyz;

    double h, hacc;
    h = dt;
    hacc = 0.0;
    double B[] = {0.25, 3.0 / 32.0, 9.0 / 32.0, 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0};
    double C[] = {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4101.0, -1.0 / 5.0, 0.0, 16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};
    do
    {
        parms.dt = h;
        ButcherTableau(6, B, C, h, K, auxT, auxQ, auxT_5, auxQ_5, T, Q, Lxyz, Lxy, parms);

        double alpha = sqrt(sqrt(TOL8_CPU / (2.0 * sqrt(MathCPU::Distance(aux, aux_5, L)) + 1e-3 * TOL8_CPU)));
        if (alpha > 1.0 || h <= 1e-1 * dt)
        {
            hacc += h;
            h = dt - hacc;
            MathCPU::Copy(T, auxT, Lxyz);
            MathCPU::Copy(Q, auxQ, Lxy);
        }
        else
        {
            h = std::max(1e-1 * dt, alpha * h);
        }
    } while (dt - hacc > 0);
    parms.dt = dt;
}

void HC::RM::CPU::EvolutionJacobianMatrix(double *pmTT_o, double *pmTQ_o, double *pmQT_o, double *pmQQ_o, HCParms &parms)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;
    int L = Lxy + Lxy;
    double dx = parms.dx;
    double dy = parms.dy;
    double dz = parms.dz;
    double amp = parms.amp;
    double T_ref = parms.T_ref;

    double CT = C(T_ref);
    double KT = K(T_ref);

    // Difusion Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {

            index = j * Lx + i;
            double aux = 0.0;
            if (i != 0)
            {
                aux += KT / (CT * dx * dx);
                pmTT_o[(index - 1) * L + index] = KT / (CT * dx * dx);
            }
            if (i != Lx - 1)
            {
                aux += KT / (CT * dx * dx);
                pmTT_o[(index + 1) * L + index] = KT / (CT * dx * dx);
            }
            if (j != 0)
            {
                aux += KT / (CT * dy * dy);
                pmTT_o[(index - Lx) * L + index] = KT / (CT * dy * dy);
            }
            if (j != Ly - 1)
            {
                aux += KT / (CT * dy * dy);
                pmTT_o[(index + Lx) * L + index] = KT / (CT * dy * dy);
            }
            pmTT_o[index * L + index] = -aux;
        }
    }

    // Heat Flux Contribution
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            index = j * Lx + i;
            pmQT_o[index * L + index] = amp / (dz * CT);
        }
    }
}

void HC::RM::CPU::EvaluationMatrix(double *pmTT_o, double *pmQT_o, HCParms &parms)
{
    int index;
    int Lx = parms.Lx;
    int Ly = parms.Ly;
    int Lxy = parms.Lx * parms.Ly;

    double KT = K(parms.T_ref);

    // Surface Temperature
    for (int j = 0; j < Ly; j++)
    {
        for (int i = 0; i < Lx; ++i)
        {
            index = j * Lx + i;
            pmTT_o[index * Lxy + index] = 1;
#if ILSA == 1
            pmQT_o[index * Lxy + index] = -parms.dz * parms.amp / (6 * KT);
#endif
        }
    }
}

__global__ void Diffusion(double *dT, double *T, double dx, double dy, double dz, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = (zIndex * Ly + yIndex) * Lx + xIndex;
    extern __shared__ double Ts[];
    unsigned tStrideX = 1u;
    unsigned tStrideY = (blockDim.x + 2u);
    unsigned tStrideZ = (blockDim.x + 2u) * (blockDim.y + 2u);
    unsigned thread = ((threadIdx.z + 1u) * (blockDim.y + 2) + (threadIdx.y + 1u)) * (blockDim.x + 2u) + (threadIdx.x + 1u);
    bool inside = xIndex < Lx && yIndex < Ly && zIndex < Lz;
    double temp = 0.0;
    unsigned begin, end;
    if (inside)
    {
        temp = T[index];
    }
    Ts[thread] = temp;
    __syncthreads();

    begin = 0u;
    end = min(Lx, (blockIdx.x + 1u) * blockDim.x) - blockIdx.x * blockDim.x - 1u;
    if (threadIdx.x == begin)
    {
        Ts[thread - tStrideX] = (xIndex == 0) ? temp : T[index - 1u];
    }
    if (threadIdx.x == end)
    {
        Ts[thread + tStrideX] = (xIndex == Lx - 1) ? temp : T[index + 1u];
    }

    begin = 0u;
    end = min(Ly, (blockIdx.y + 1u) * blockDim.y) - blockIdx.y * blockDim.y - 1u;
    if (threadIdx.y == begin)
    {
        Ts[thread - tStrideY] = (yIndex == 0) ? temp : T[index - Lx];
    }
    if (threadIdx.y == min(Ly - blockIdx.y * blockDim.y, blockDim.y) - 1)
    {
        Ts[thread + tStrideY] = (yIndex == Ly - 1) ? temp : T[index + Lx];
    }

    begin = 0u;
    end = min(Lz, (blockIdx.z + 1u) * blockDim.z) - blockIdx.z * blockDim.z - 1u;
    if (threadIdx.z == begin)
    {
        Ts[thread - tStrideZ] = (zIndex == 0) ? temp : T[index - Lx * Ly];
    }
    if (threadIdx.z == end)
    {
        Ts[thread + tStrideZ] = (zIndex == Lz - 1) ? temp : T[index + Lx * Ly];
    }
    __syncthreads();

    double aux = DiffK(Ts[thread - tStrideX], Ts[thread], Ts[thread + tStrideX], dx) + DiffK(Ts[thread - tStrideY], Ts[thread], Ts[thread + tStrideY], dy) + DiffK(Ts[thread - tStrideZ], Ts[thread], Ts[thread + tStrideZ], dz);

    if (inside)
    {
        dT[index] = aux;
    }
}
__global__ void HeatFluxContribution(double *dT, double *Q, double amp, double dz, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned index = yIndex * Lx + xIndex;
    bool inside = xIndex < Lx && yIndex < Ly;
    unsigned offset = Lx * Ly * (Lz - 1u);
    if (inside)
    {
        dT[offset + index] += (amp / dz) * Q[index];
    }
}
__global__ void ThermalCapacity(double *dT, double *T, unsigned Lx, unsigned Ly, unsigned Lz)
{
    unsigned xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned zIndex = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned index = (zIndex * Ly + yIndex) * Lx + xIndex;
    bool inside = xIndex < Lx && yIndex < Ly && zIndex < Lz;
    if (inside)
    {
        dT[index] /= HC::C(T[index]);
    }
}
void HC::GPU::Diff(double *dT, double *dQ, double *T, double *Q, HCParms &parms)
{
    dim3 T3(16, 16, 4);
    dim3 T2(32, 32);
    dim3 B3(CEIL(parms.Lx, 16), CEIL(parms.Ly, 16), CEIL(parms.Lz, 4));
    dim3 B2(CEIL(parms.Lx, 32), CEIL(parms.Ly, 32));
    MathGPU::Zero(dT, parms.Lx * parms.Ly * parms.Lz);
    Diffusion<<<B3, T3, sizeof(double) * (T3.x + 2) * (T3.y + 2) * (T3.z + 2), cudaStreamDefault>>>(dT, T, parms.dx, parms.dy, parms.dz, parms.Lx, parms.Ly, parms.Lz);
    HeatFluxContribution<<<B2, T2, 0, cudaStreamDefault>>>(dT, Q, parms.amp, parms.dz, parms.Lx, parms.Ly, parms.Lz);
    ThermalCapacity<<<B3, T3, 0, cudaStreamDefault>>>(dT, T, parms.Lx, parms.Ly, parms.Lz);
    MathGPU::Zero(dQ, parms.Lx * parms.Ly);
}
void HC::GPU::AllocWorkspaceEuler(double *&workspace, HCParms &parms)
{
    cudaMallocAsync(&workspace, sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1), cudaStreamDefault);
}
void HC::GPU::FreeWorkspaceEuler(double *workspace)
{
    cudaFreeAsync(workspace, cudaStreamDefault);
}
void HC::GPU::Euler(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    double dt = parms.dt;
    double *dT = workspace;
    double *dQ = workspace + Lxyz;

    Diff(dT, dQ, T, Q, parms);
    MathGPU::Mul(dT, dt, Lxyz);
    MathGPU::Mul(dQ, dt, Lxy);
    MathGPU::Add(T, dT, Lxyz);
    MathGPU::Add(Q, dQ, Lxy);
}
void HC::GPU::AllocWorkspaceRK4(double *&workspace, HCParms &parms)
{
    cudaMallocAsync(&workspace, 5 * sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1), cudaStreamDefault);
}
void HC::GPU::FreeWorkspaceRK4(double *workspace)
{
    cudaFreeAsync(workspace, cudaStreamDefault);
}
void HC::GPU::RK4(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    int L = Lxyz + Lxy;
    double dt = parms.dt;
    double *K1, *K2, *K3, *K4, *aux, *auxT, *auxQ;
    K1 = workspace;
    K2 = K1 + L;
    K3 = K2 + L;
    K4 = K3 + L;
    aux = K4 + L;
    auxT = aux;
    auxQ = auxT + Lxyz;

    Diff(K1, K1 + Lxyz, T, Q, parms);

    MathGPU::Mul(aux, K1, 0.5 * dt, L);
    MathGPU::Add(auxT, T, Lxyz);
    MathGPU::Add(auxQ, Q, Lxy);
    Diff(K2, K2 + Lxyz, auxT, auxQ, parms);

    MathGPU::Mul(aux, K2, 0.5 * dt, L);
    MathGPU::Add(auxT, T, Lxyz);
    MathGPU::Add(auxQ, Q, Lxy);
    Diff(K3, K3 + Lxyz, auxT, auxQ, parms);

    MathGPU::Mul(aux, K3, dt, L);
    MathGPU::Add(auxT, T, Lxyz);
    MathGPU::Add(auxQ, Q, Lxy);
    Diff(K4, K4 + Lxyz, auxT, auxQ, parms);

    MathGPU::Add(K1, K4, L);
    MathGPU::Add(K2, K3, L);
    MathGPU::Mul(K1, dt / 6.0, L);
    MathGPU::Mul(K2, dt / 3.0, L);
    MathGPU::Add(K1, K2, L);
    MathGPU::Add(T, K1, Lxyz);
    MathGPU::Add(Q, K1 + Lxyz, Lxy);
    cudaDeviceSynchronize();
}
void HC::GPU::AllocWorkspaceRKF45(double *&workspace, HCParms &parms)
{
    cudaMallocAsync(&workspace, 8 * sizeof(double) * parms.Lx * parms.Ly * (parms.Lz + 1), cudaStreamDefault);
}
void HC::GPU::FreeWorkspaceRKF45(double *workspace)
{
    cudaFreeAsync(workspace, cudaStreamDefault);
}
void HC::GPU::ButcherTableau(int n, double *B, double *C, double h, double *K, double *aux00, double *aux01, double *aux10, double *aux11, double *ref0, double *ref1, int L0, int L1, HCParms &parms)
{
    int L = L0 + L1;
    for (int d = 0; d < n; ++d)
    {
        double *K_d = K + L * d;
        MathGPU::Copy(aux00, ref0, L0);
        MathGPU::Copy(aux01, ref1, L1);
        for (int j = 0; j < d; ++j)
        {
            double *K_j = K + L * j;
            MathGPU::LRPO(aux00, K_j, B[bI(d) + j] * h, L0);
            MathGPU::LRPO(aux01, K_j + L0, B[bI(d) + j] * h, L1);
        }
        HC::GPU::Diff(K_d, K_d + L0, aux00, aux01, parms);
    }
    MathGPU::Copy(aux00, ref0, L0);
    MathGPU::Copy(aux10, ref0, L0);
    MathGPU::Copy(aux01, ref1, L1);
    MathGPU::Copy(aux11, ref1, L1);
    for (int j = 0; j < n; ++j)
    {
        double *K_j = K + L * j;
        double temp = C[j] * h;
        MathGPU::LRPO(aux00, K_j, C[j] * h, L0);
        MathGPU::LRPO(aux01, K_j + L0, C[j] * h, L1);
        MathGPU::LRPO(aux10, K_j, C[n + j] * h, L0);
        MathGPU::LRPO(aux11, K_j + L0, C[n + j] * h, L1);
    }
}
void HC::GPU::RKF45(double *T, double *Q, double *workspace, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;
    int Lxyz = Lxy * parms.Lz;
    int L = Lxyz + Lxy;
    double dt = parms.dt;
    double *K, *aux, *auxT, *auxQ, *aux_5, *auxT_5, *auxQ_5;
    K = workspace;
    aux = K + 6 * L;
    auxT = aux;
    auxQ = auxT + Lxyz;
    aux_5 = aux + L;
    auxT_5 = auxQ + Lxy;
    auxQ_5 = auxT_5 + Lxyz;

    double h = dt;
    double hacc = 0.0;
    double B[] = {0.25, 3.0 / 32.0, 9.0 / 32.0, 1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0, 439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0, -8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0};
    double C[] = {25.0 / 216.0, 0.0, 1408.0 / 2565.0, 2197.0 / 4101.0, -1.0 / 5.0, 0.0, 16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0};

    do
    {
        parms.dt = dt;
        ButcherTableau(6, B, C, h, K, auxT, auxQ, auxT_5, auxQ_5, T, Q, Lxyz, Lxy, parms);

        double alpha = sqrt(sqrt(TOL8_CPU / (2.0 * sqrt(MathGPU::Distance(aux, aux_5, L)) + 1e-3 * TOL8_CPU)));
        if (alpha > 1.0 || h <= 1e-1 * dt)
        {
            hacc += h;
            h = dt - hacc;
            MathGPU::Copy(T, auxT, Lxyz);
            MathGPU::Copy(Q, auxQ, Lxy);
        }
        else
        {
            h = std::max(1e-1 * dt, alpha * h);
        }
    } while (dt - hacc);
}

void HC::RM::GPU::EvolutionJacobianMatrix(double *pmTT_o, double *pmTQ_o, double *pmQT_o, double *pmQQ_o, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;

    double *pm, *pm_o, *pmTT, *pmTQ, *pmQT, *pmQQ;

    int stride11 = pmTQ_o - pmTT_o;
    int stride22 = pmQQ_o - pmTT_o;

    pm = (double *)malloc(sizeof(double) * 4 * Lxy * Lxy);
    for (int i = 0; i < 4 * Lxy * Lxy; i++)
    {
        pm[i] = 0.0;
    }

    pmTT = pm + std::max(-stride22, 0);
    pmTQ = pm + std::max(-stride22, 0) + std::max(stride11, 0);
    pmQT = pm + std::max(stride22, 0) + std::max(-stride11, 0);
    pmQQ = pm + std::max(stride22, 0);

    HC::RM::CPU::EvolutionJacobianMatrix(pmTT, pmTQ, pmQT, pmQQ, parms);

    pm_o = std::min(pmTT_o, pmQQ_o);
    cudaMemcpy(pm_o, pm, 4 * sizeof(double) * Lxy * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}

void HC::RM::GPU::EvaluationMatrix(double *pmTT_o, double *pmQT_o, HCParms &parms)
{
    int Lxy = parms.Lx * parms.Ly;

    double *pm, *pm_o, *pmTT, *pmQT;

    int stride12 = pmQT_o - pmTT_o;

    pm = (double *)malloc(sizeof(double) * 2 * Lxy * Lxy);
    for (int i = 0; i < 2 * Lxy * Lxy; i++)
    {
        pm[i] = 0.0;
    }

    pmTT = pm + std::max(-stride12, 0);
    pmQT = pm + std::max(stride12, 0);

    HC::RM::CPU::EvaluationMatrix(pmTT, pmQT, parms);

    pm_o = std::min(pmTT_o, pmQT_o);
    cudaMemcpy(pm_o, pm, 2 * sizeof(double) * Lxy * Lxy, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    free(pm);
}