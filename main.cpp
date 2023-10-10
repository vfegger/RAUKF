#include "./src/hfe/include/hfe.hpp"
#include "./src/hfe/include/hcr.hpp"
#include "./src/filter/include/raukf.hpp"
#include "./src/filter/include/kf.hpp"

#include <fstream>
#include <random>

void Simulation(double *measures, int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp)
{
    double *workspace;
    HCR::HCRParms parms;
    parms.Lx = Lx;
    parms.Ly = Ly;
    parms.Lz = Lz;
    parms.Lt = Lt;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lx;
    parms.dy = Sy / Ly;
    parms.dz = Sz / Lz;
    parms.dt = St / Lt;
    parms.amp = amp;
    double *T = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *Q = (double *)malloc(sizeof(double) * Lx * Ly);
    for (int i = 0; i < Lx * Ly * Lz; ++i)
    {
        T[i] = 300.0;
    }
    MathCPU::Zero(Q, Lx * Ly);
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            bool xCond = (i + 0.5) * Sx / Lx > 0.3 * Sx && (i + 0.5) * Sx / Lx < 0.7 * Sx;
            Q[j * Lx + i] = (xCond) ? 100.0 : 0.0;
        }
    }
    HCR::CPU::AllocWorkspaceRKF45(workspace, parms);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 5.0);
    for (int i = 0; i < Lt; ++i)
    {
        HCR::CPU::RKF45(T, Q, workspace, parms);
        for (int j = 0; j < Lx * Ly; ++j)
        {
            measures[i * Lx * Ly + j] = T[j] + distribution(generator);
        }
    }
    HCR::CPU::FreeWorkspaceRKF45(workspace);
    free(Q);
    free(T);
}

int main(int argc, char *argv[])
{
    Type type = Type::CPU;
    if (argc > 1)
    {
        type = (std::stoi(argv[1]) == 0) ? Type::CPU : Type::GPU;
    }
    if (type == Type::GPU)
    {
        cudaDeviceReset();
        MathGPU::CreateHandles();
    }
    int Lx = 24;
    int Ly = 24;
    int Lz = 6;
    int Lt = 100;
    double Sx = 0.12;
    double Sy = 0.12;
    double Sz = 0.003;
    double St = 2.0;
    double amp = 5e4;

    Timer timer;
    HFE hfe;
    HFE hfeKF;
    RAUKF raukf;
    KF kf;

    hfe.SetParms(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp);
    hfe.SetMemory(type);
    hfeKF.SetParms(Lx, Ly, 1, Lt, Sx, Sy, Sz, St, amp);
    hfeKF.SetMemory(type);

    raukf.SetParameters(1e-3, 2.0, 0.0);
    raukf.SetModel(&hfe);
    raukf.SetType(type);
    raukf.SetWeight();

    kf.SetModel(&hfeKF);
    kf.SetType(type);

    double *measures = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    Simulation(measures, Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp);
    double *resultT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultCovarQ = (double *)malloc(sizeof(double) * Lx * Ly);

    for (int i = 0; i < Lt; i++)
    {
        raukf.SetMeasure("Temperature", measures + Lx * Ly * i);
        kf.SetMeasure("Temperature", measures + Lx * Ly * i);

        raukf.Iterate(timer);
        raukf.GetState("Temperature", resultT);
        raukf.GetState("Heat Flux", resultQ);
        raukf.GetStateCovariance("Temperature", resultCovarT);
        raukf.GetStateCovariance("Heat Flux", resultCovarQ);

        std::ofstream outFile;
        outFile.open("data/raukf/Values" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
        if (outFile.is_open())
        {
            double resultTime = (i + 1) * St / Lt;
            outFile.write((char *)(&resultTime), sizeof(double));
            outFile.write((char *)resultT, sizeof(double) * Lx * Ly * Lz);
            outFile.write((char *)resultCovarT, sizeof(double) * Lx * Ly * Lz);
            outFile.write((char *)resultQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarQ, sizeof(double) * Lx * Ly);
        }
        outFile.close();

        outFile.open("data/raukf/ready/" + std::to_string(i), std::ios::out | std::ios::binary);
        outFile.close();

        kf.Iterate(timer);
        kf.GetMeasure("Temperature", resultT);
        kf.GetState("Heat Flux", resultQ);
        kf.GetStateCovariance("Temperature", resultCovarT);
        kf.GetStateCovariance("Heat Flux", resultCovarQ);

        outFile.open("data/kf/Values" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
        if (outFile.is_open())
        {
            double resultTime = (i + 1) * St / Lt;
            outFile.write((char *)(&resultTime), sizeof(double));
            outFile.write((char *)resultT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarQ, sizeof(double) * Lx * Ly);
        }
        outFile.close();

        outFile.open("data/kf/ready/" + std::to_string(i), std::ios::out | std::ios::binary);
        outFile.close();
    }

    free(resultCovarQ);
    free(resultCovarT);
    free(resultQ);
    free(resultT);
    free(measures);

    raukf.UnsetWeight();
    raukf.UnsetModel();
    hfe.UnsetMemory(type);
    if (type == Type::GPU)
    {
        MathGPU::DestroyHandles();
        cudaDeviceReset();
    }
    return 0;
}