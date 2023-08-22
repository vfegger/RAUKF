#include "./src/hfe/include/hfe.hpp"
#include "./src/hfe/include/hcr.hpp"
#include "./src/raukf/include/raukf.hpp"

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
    HCR::CPU::AllocWorkspaceRKF45(workspace, parms);
    for (int i = 0; i < Lt; ++i)
    {
        HCR::CPU::Euler(T, Q, workspace, parms);
        for (int j = 0; j < Lx * Ly; ++j)
        {
            measures[i * Lx * Ly + j] = T[j];
        }
    }
    HCR::CPU::FreeWorkspaceRKF45(workspace);
}

int main()
{
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
    RAUKF raukf;

    hfe.SetParms(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp);
    hfe.SetMemory(Type::CPU);

    raukf.SetParameters(1e-3, 2.0, 0.0);
    raukf.SetModel(&hfe);
    raukf.SetType(Type::CPU);

    raukf.SetWeight();

    double *measures = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    Simulation(measures, Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp);
    double *resultT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultCovarQ = (double *)malloc(sizeof(double) * Lx * Ly);

    for (int i = 0; i < Lt; i++)
    {
        raukf.SetMeasure("Temperature", measures + Lx * Ly * i);
        raukf.Iterate(timer);
        raukf.GetState("Temperature", resultT);
        raukf.GetState("Heat Flux", resultQ);
        raukf.GetState("Temperature", resultCovarT);
        raukf.GetState("Heat Flux", resultCovarQ);
    }

    raukf.UnsetModel();
    hfe.UnsetMemory(Type::CPU);
    return 0;
}