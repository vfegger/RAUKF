#include "./src/math/include/interpolation.hpp"
#include "./src/hfe2D/include/hfe2D.hpp"
#include "./src/hfe2D/include/hc2D.hpp"
#include "./src/filter/include/kf.hpp"
#include "./src/filter/include/raukf.hpp"

#include <fstream>
#include <random>
#include <string>
#include <format>

#define RAUKF_USAGE 0
#define KF_USAGE 1
#define NOISE_USAGE 1

#define LX_DEFAULT (32)
#define LY_DEFAULT (32)
#define LT_DEFAULT (500 * 12)

#define SIMULATION_CASE 0

void AddNoise(std::default_random_engine &gen, std::normal_distribution<double> &dist, double *v_o, double *v_i, int length)
{
    for (int i = 0; i < length; i++)
    {
#if NOISE_USAGE == 1
        v_o[i] = v_i[i] + dist(gen);
#else
        v_o[i] = v_i[i];
#endif
    }
}

void CaseHeatFlux(double *Qh, double t, int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp)
{
    double p = 50;
    double a = 1.364e-2;
    double b = 0.922e-2;
    double c = 0.330e-2;
#if SIMULATION_CASE == 0
    double Sx1 = 0.4 * Sx;
    double Sx2 = 0.6 * Sx;
    double Sy1 = 0.4 * Sy;
    double Sy2 = 0.6 * Sy;
    double St1 = 50.0;
    double St2 = 150.0;
    double w = (p / ((Sx2 - Sx1) * (Sy2 - Sy1))) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool xCond = (xi > Sx1 && xi < Sx2);
            bool yCond = (yj > Sy1 && yj < Sy2);
            bool tCond = t > St1 && t < St2;
            Qh[j * Lx + i] = (xCond && yCond && tCond) ? w : 0.0;
        }
    }
#elif SIMULATION_CASE == 1
    double Sx1 = 0.5 * Sx;
    double Sy1 = 0.5 * Sx;
    double Sr1 = 0.1 * std::min(Sx, Sy);
    double St1 = 50.0;
    double St2 = 150.0;
    double w = (p / (M_PI * Sr1 * Sr1)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool rCond = (xi - Sx1) * (xi - Sx1) + (yj - Sy1) * (yj - Sy1) < Sr1 * Sr1;
            bool tCond = t > St1 && t < St2;
            Qh[j * Lx + i] = (rCond && tCond) ? w : 0.0;
        }
    }
#elif SIMULATION_CASE == 2
    double Sx1 = (Sx - c) / 2.0 - b;
    double Sx2 = (Sx + c) / 2.0;
    double Sy1 = (Sy - a) / 2.0;
    double St1 = 50.0;
    double St2 = 150.0;
    double w = (p / (2 * a * b)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool xCond = (xi > Sx1 && xi < Sx1 + b) || (xi > Sx2 && xi < Sx2 + b);
            bool yCond = (yj > Sy1 && yj < Sy1 + a);
            bool tCond = t > St1 && t < St2;
            Qh[j * Lx + i] = (xCond && yCond && tCond) ? w : 0.0;
        }
    }
#elif SIMULATION_CASE == 3
    double Sx1 = 0.3 * Sx;
    double Sx2 = 0.4 * Sx;
    double Sy1 = 0.3 * Sy;
    double Sy2 = 0.4 * Sy;
    double St1 = 50.0;
    double St2 = 150.0;
    double w = (p / (2 * a * b)) / amp;
    MathCPU::Zero(Qh, Lx * Ly);

    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool xCond = (xi > Sx1 && xi < Sx2);
            bool yCond = (yj > Sy1 && yj < Sy2);
            bool tCond = t > St1 && t < St2;
            if (xCond && yCond && tCond)
            {
                Qh[j * Lx + i] += w;
            }
        }
    }


    double Sx1 = 0.5 * Sx;
    double Sy1 = 0.5 * Sx;
    double Sr1 = 0.1 * std::min(Sx, Sy);
    double St1 = St * 0.01;
    double St2 = St * 0.02;
    double w = (p / (M_PI * Sr1 * Sr1)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool rCond = (xi - Sx1) * (xi - Sx1) + (yj - Sy1) * (yj - Sy1) < Sr1 * Sr1;
            bool tCond = t > St1 && t < St2;
            if (rCond && tCond)
            {
                Qh[j * Lx + i] += w;
            }
        }
    }

#endif
}

void Simulation(double *measures, double *Q_ref, int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double h, Type type)
{
    std::cout << "Generating synthetic measurements.\n";
    double *workspace;
    HC2D::HCParms parms;
    int Lsx = Lx;
    int Lsy = Ly;
    int Lxy = Lx * Ly;
    int Lsxy = Lsx * Lsy;
    int n = 10;
    int Lst = n * Lt;
    int L = 2 * Lsxy;
    int Lu = 1;
    parms.Lx = Lsx;
    parms.Ly = Lsy;
    parms.Lt = Lst;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lsx;
    parms.dy = Sy / Lsy;
    parms.dt = St / Lst;
    parms.amp = amp;
    parms.h = h;

    Pointer<double> X, T, Q, aux;
    Pointer<double> U, T_amb;
    Pointer<double> Y, T_m;
    Pointer<double> XX, UX, XY;
    Pointer<double> Tr, Qr;
    X.alloc(L);
    T = X + Lsxy;
    Q = X;
    aux.alloc(L);

    U.alloc(Lu);
    T_amb = U;

    Y.alloc(Lsxy);
    T_m = Y;

    XX.alloc(L * L);
    UX.alloc(L * Lu);
    XY.alloc(L * Lsxy);

    Tr.alloc(Lt * Lxy);
    Qr.alloc(Lt * Lxy);
    Math::Zero(T, Lsxy, Type::CPU);
    Math::Zero(Q, Lsxy, Type::CPU);
    Math::Zero(T_m, Lsxy, Type::CPU);

    double *Th = T.host();
    for (int i = 0; i < Lsxy; ++i)
    {
        Th[i] = 300.0;
    }
    double *T_ambh = T_amb.host();
    T_ambh[0] = 300.0;
    if (type == Type::GPU)
    {
        T.copyHost2Dev(Lsxy);
        Q.copyHost2Dev(Lsxy);
        U.copyHost2Dev(Lu);
        T_m.copyHost2Dev(Lsxy);
        cudaDeviceSynchronize();
    }
    // Invariant Evolution
    if (type == Type::GPU)
    {
        HC2D::GPU::EvolutionMatrix(parms, XX.dev(), UX.dev(), (Q.dev() - T.dev()));
        XX.copyDev2Host(L * L);
        HC2D::GPU::EvaluationMatrix(parms, XY.dev(), (Q.dev() - T.dev()));
        XY.copyDev2Host(L * Lsxy);
        cudaDeviceSynchronize();
    }
    else
    {
        HC2D::CPU::EvolutionMatrix(parms, XX.host(), UX.host(), (Q.host() - T.host()));
        HC2D::GPU::EvaluationMatrix(parms, XY.host(), (Q.host() - T.host()));
    }

    for (int k = 0; k < Lst; ++k)
    {
        // Heat Flux Definition
        CaseHeatFlux(Q.host(), (k + 1) * St / Lst, Lsx, Lsy, Lst, Sx, Sy, Sz, St, amp);
        if (type == Type::GPU)
        {
            Q.copyHost2Dev(Lsx * Lsy);
        }
        Math::Copy(aux, X, L, type);
        Math::MatMulNN(0.0, X, 1.0, XX, aux, L, L, 1, type);
        Math::MatMulNN(1.0, X, 1.0, UX, U, L, Lu, 1, type);
        cudaDeviceSynchronize();
        Math::MatMulNN(0.0, Y, 1.0, XY, X, Lsxy, L, 1, type);
        cudaDeviceSynchronize();
        if (k % n == n - 1)
        {
            Interpolation::Rescale(T_m, Lsx, Lsy, Tr + (k / n) * Lx * Ly, Lx, Ly, Sx, Sy, type);
            Interpolation::Rescale(Q, Lsx, Lsy, Qr + (k / n) * Lx * Ly, Lx, Ly, Sx, Sy, type);
        }
        std::cout << "Iteration: " << k + 1 << "/" << Lst << "\r" << std::flush;
    }

    if (type == Type::GPU)
    {
        Tr.copyDev2Host(Lt * Lx * Ly);
        Qr.copyDev2Host(Lt * Lx * Ly);
        cudaDeviceSynchronize();
    }
    std::cout << "\n";
    MathCPU::Copy(measures, Tr.host(), Lx * Ly * Lt);
    for (int k = 0; k < Lt; ++k)
    {
        std::cout << std::format("{:.8f}", measures[k * Lx * Ly + Lx * (Ly + 1) / 2]) << "\n";
    }
    MathCPU::Copy(Q_ref, Qr.host(), Lx * Ly * Lt);
    Qr.free();
    Tr.free();
    XY.free();
    UX.free();
    XX.free();
    Y.free();
    U.free();
    X.free();
    std::cout << "Synthetic measurements generated.\n";
}

void ReadMeasurements(double *measures, double *Q_ref, int Lx, int Ly, int Lt)
{
    std::ifstream inFile;
    int Lxy = Lx * Ly;
    for (int i = 0; i < Lt; i++)
    {
        inFile.open("input/Values" + std::to_string(i) + ".bin", std::ios::in | std::ios::binary);
        if (inFile.is_open())
        {
            inFile.read((char *)(measures + i * Lxy), sizeof(double) * Lxy);
        }
        inFile.close();
    }
    MathCPU::Zero(Q_ref, Lx * Ly * Lt);
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
    int Lx = LX_DEFAULT;
    int Ly = LY_DEFAULT;
    int Lt = LT_DEFAULT;
    double Sx = 0.0296;
    double Sy = 0.0296;
    double Sz = 0.0015;
    double St = 500.0;
    double amp = 1.0e3;
    double h = 0.0; // 11.0;

    std::ofstream outParms;
    int temp;
#if RAUKF_USAGE == 1
    outParms.open("data/raukf/Parms.bin", std::ios::out | std::ios::binary);
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();
#endif
#if KF_USAGE == 1
    outParms.open("data/kf/Parms.bin", std::ios::out | std::ios::binary);
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();
#endif

    Timer timer;
#if RAUKF_USAGE == 1
    HFE2D hfe;
#endif
#if KF_USAGE == 1
    HFE2D hfeKF;
#endif

#if RAUKF_USAGE == 1
    RAUKF raukf;
#endif
#if KF_USAGE == 1
    KF kf;
#endif

#if RAUKF_USAGE == 1
    hfe.SetParms(Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h);
    hfe.SetMemory(type);
#endif
#if KF_USAGE == 1
    hfeKF.SetParms(Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h);
    hfeKF.SetMemory(type);
#endif

#if RAUKF_USAGE == 1
    raukf.SetParameters(1e-3, 2.0, 0.0);
    raukf.SetModel(&hfe);
    raukf.SetType(type);
    raukf.SetWeight();
#endif
#if KF_USAGE == 1
    kf.SetModel(&hfeKF);
    kf.SetType(type);
#endif

    double *measures = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *measuresN = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *q_ref = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
#if USE_MEASUREMENTS == 1
    ReadMeasurements(measures, q_ref, Lx, Ly, Lt);
#else
    Simulation(measures, q_ref, Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h, type);
#endif
    double *resultT = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarT = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultTm = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarTm = (double *)malloc(sizeof(double) * Lx * Ly);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    AddNoise(generator, distribution, measuresN, measures, Lx * Ly * Lt);

    std::ofstream outFile;
    for (int i = 0; i < Lt; i++)
    {
#if RAUKF_USAGE == 1
        raukf.SetMeasure("Temperature", measuresN + Lx * Ly * i);

        raukf.Iterate(timer);
        raukf.GetState("Temperature", resultT);
        raukf.GetState("Heat Flux", resultQ);
        raukf.GetStateCovariance("Temperature", resultCovarT);
        raukf.GetStateCovariance("Heat Flux", resultCovarQ);
        raukf.GetMeasure("Temperature", resultTm);
        raukf.GetMeasureCovariance("Temperature", resultCovarTm);

        outFile.open("data/raukf/Values" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
        if (outFile.is_open())
        {
            double resultTime = (i + 1) * St / Lt;
            outFile.write((char *)(&resultTime), sizeof(double));
            outFile.write((char *)resultT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultTm, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarTm, sizeof(double) * Lx * Ly);
            outFile.write((char *)(measuresN + Lx * Ly * i), sizeof(double) * Lx * Ly);
            outFile.write((char *)(measures + Lx * Ly * i), sizeof(double) * Lx * Ly);
#if USE_MEASUREMENTS == 0
            outFile.write((char *)(q_ref + Lx * Ly * i), sizeof(double) * Lx * Ly);
#endif
        }
        outFile.close();

        outFile.open("data/raukf/ready/" + std::to_string(i), std::ios::out | std::ios::binary);
        outFile.close();
#endif

#if KF_USAGE == 1
        kf.SetMeasure("Temperature", measuresN + Lx * Ly * i);

        kf.Iterate(timer);
        kf.GetState("Temperature", resultT);
        kf.GetState("Heat Flux", resultQ);
        kf.GetStateCovariance("Temperature", resultCovarT);
        kf.GetStateCovariance("Heat Flux", resultCovarQ);
        kf.GetMeasure("Temperature", resultTm);
        kf.GetMeasureCovariance("Temperature", resultCovarTm);

        outFile.open("data/kf/Values" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
        if (outFile.is_open())
        {
            double resultTime = (i + 1) * St / Lt;
            outFile.write((char *)(&resultTime), sizeof(double));
            outFile.write((char *)resultT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarT, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarQ, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultTm, sizeof(double) * Lx * Ly);
            outFile.write((char *)resultCovarTm, sizeof(double) * Lx * Ly);
            outFile.write((char *)(measuresN + Lx * Ly * i), sizeof(double) * Lx * Ly);
            outFile.write((char *)(measures + Lx * Ly * i), sizeof(double) * Lx * Ly);
            outFile.write((char *)(q_ref + Lx * Ly * i), sizeof(double) * Lx * Ly);
        }
        outFile.close();

        outFile.open("data/kf/ready/" + std::to_string(i), std::ios::out | std::ios::binary);
        outFile.close();
#endif
    }

    free(resultCovarTm);
    free(resultTm);
    free(resultCovarQ);
    free(resultCovarT);
    free(resultQ);
    free(resultT);
#if USE_MEASUREMENTS == 0
    free(q_ref);
#endif
    free(measuresN);
    free(measures);

#if KF_USAGE == 1
    kf.UnsetModel();
#endif
#if RAUKF_USAGE == 1
    raukf.UnsetWeight();
    raukf.UnsetModel();
#endif
#if KF_USAGE == 1
    hfeKF.UnsetMemory(type);
#endif
#if RAUKF_USAGE == 1
    hfe.UnsetMemory(type);
#endif

    if (type == Type::GPU)
    {
        MathGPU::DestroyHandles();
        cudaDeviceReset();
    }
    return 0;
}