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

void Simulation(double *measures, double *Q_ref, int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double h, Type type)
{
    std::cout << "Generating synthetic measurements.\n";
    double *workspace;
    HC2D::HCParms parms;
    int Lsx = Lx;
    int Lsy = Ly;
    int n = 2000;
    int Lst = n * Lt;
    int L = 2 * Lsx * Lsy;
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
    Pointer<double> XX, TT, TQ, QT, QQ;
    Pointer<double> UX, UT, UQ;
    Pointer<double> Tr, Qr;
    X.alloc(L);
    T = X;
    Q = X + Lsx * Lsy;
    aux.alloc(L);

    U.alloc(Lu);
    T_amb = U;

    XX.alloc(L * L);
    TT = XX;
    TQ = XX + Lsx * Lsy;
    QT = XX + Lsx * Lsy * L;
    QQ = XX + Lsx * Lsy * L + Lsx * Lsy;

    UX.alloc(L * Lu);
    UT = UX;
    UQ = UX + Lsx * Lsy;

    Tr.alloc(Lt * Lx * Ly);
    Qr.alloc(Lt * Lx * Ly);
    Math::Zero(T, Lsx * Lsy, Type::CPU);
    Math::Zero(Q, Lsx * Lsy, Type::CPU);

    double *Th = T.host();
    for (int i = 0; i < Lsx * Lsy; ++i)
    {
        Th[i] = 300.0;
    }
    double *T_ambh = T_amb.host();
    T_ambh[0] = 300.0;
    if (type == Type::GPU)
    {
        T.copyHost2Dev(Lsx * Lsy);
        Q.copyHost2Dev(Lsx * Lsy);
        U.copyHost2Dev(Lu);
        cudaDeviceSynchronize();
    }
    // Invariant Evolution
    if (type == Type::GPU)
    {
        HC2D::GPU::EvolutionJacobianMatrix(TT.dev(), TQ.dev(), QT.dev(), QQ.dev(), parms);
        HC2D::GPU::EvolutionControlMatrix(UT.dev(), UQ.dev(), parms);
        XX.copyDev2Host(L * L);
        cudaDeviceSynchronize();
    }
    else
    {
        HC2D::CPU::EvolutionJacobianMatrix(TT.host(), TQ.host(), QT.host(), QQ.host(), parms);
        HC2D::CPU::EvolutionControlMatrix(UT.host(), UQ.host(), parms);
    }

    for (int i = 0; i < L; ++i)
    {
        double *XX_h = XX.host();
        std::cout << " " << std::format("{:.10f}", parms.dt * XX_h[i * L + i]);
    }

    double p = 50;
    double a = 1.364e-2;
    double b = 0.922e-2;
    double c = 0.330e-2;
    double Sx1 = (Sx - c) / 2.0 - b;
    double Sx2 = (Sx + c) / 2.0;
    double Sy1 = (Sy - a) / 2.0;
    double St1 = 0.0;
    double St2 = 20.0;
    double w = p / (2 * a * b);

    for (int k = 0; k < Lst; ++k)
    {
        // Heat Flux Definition
        double *Qh = Q.host();
        for (int j = 0; j < Lsy; ++j)
        {
            for (int i = 0; i < Lsx; ++i)
            {
                double xi = (i + 0.5) * Sx / Lsx;
                double yj = (j + 0.5) * Sy / Lsy;
                bool xCond = (xi > Sx1 && xi < Sx1 + b) || (xi > Sx2 && xi < Sx2 + b);
                bool yCond = (yj > Sy1 && yj < Sy1 + a);
                bool tCond = k * St / Lst > 0.0 && k * St / Lst < 0.1 * St;
                Qh[j * Lsx + i] = (xCond && yCond && tCond) ? w : 0.0;
            }
        }
        if (type == Type::GPU)
        {
            Q.copyHost2Dev(Lsx * Lsy);
        }
        Math::Copy(aux, X, L, type);
        Math::MatMulNN(1.0, X, parms.dt, XX, aux, L, L, 1, type);
        Math::MatMulNN(1.0, X, parms.dt, UX, U, L, Lu, 1, type);
        if (k % n == n - 1)
        {
            Interpolation::Rescale(T, Lsx, Lsy, Tr + (k / n) * Lx * Ly, Lx, Ly, Sx, Sy, type);
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
    UX.free();
    XX.free();
    U.free();
    X.free();
    std::cout << "Synthetic measurements generated.\n";
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
    int Lx = 32;
    int Ly = 32;
    int Lt = 500;
    double Sx = 0.0296;
    double Sy = 0.0296;
    double Sz = 0.0015;
    double St = 1000.0;
    double amp = 1;
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
    Simulation(measures, q_ref, Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h, type);
    double *resultT = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarT = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultTm = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarTm = (double *)malloc(sizeof(double) * Lx * Ly);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 2.0);
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
            outFile.write((char *)(q_ref + Lx * Ly * i), sizeof(double) * Lx * Ly);
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
    free(q_ref);
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