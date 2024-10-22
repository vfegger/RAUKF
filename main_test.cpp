#include "./src/math/include/interpolation.hpp"
#include "./src/hfe2D/include/hfe2D.hpp"
#include "./src/hfe2D/include/hc2D.hpp"
#include "./src/filter/include/kf.hpp"
#include "./src/filter/include/raukf.hpp"

#include <fstream>
#include <random>
#include <string>
#include <format>

#define NOISE_USAGE 0

#define LX_DEFAULT (32)
#define LY_DEFAULT (32)
#define LT_DEFAULT (100 * 30)

#define SIMULATION_CASE 3

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
    double St2 = 60.0;
    double w = (p / ((Sx2 - Sx1) * (Sy2 - Sy1))) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool tCond = t > St1 && t < St2;
            double dx2 = 0.5 * Sx / Lx;
            double dy2 = 0.5 * Sy / Ly;
            double A = std::max((std::min(Sx2, xi + dx2) - std::max(Sx1, xi - dx2)), 0.0) * std::max((std::min(Sy2, yj + dy2) - std::max(Sy1, yj - dy2)), 0.0);
            double Ac = 4.0 * dx2 * dy2;
            Qh[j * Lx + i] = (tCond) ? w * A / Ac : 0.0;
        }
    }
#elif SIMULATION_CASE == 1
    double Sx1 = 0.5 * Sx;
    double Sy1 = 0.5 * Sx;
    double Sr1 = 0.1 * std::min(Sx, Sy);
    double St1 = 50.0;
    double St2 = 60.0;
    double w = (p / (1.6e-3 * Sx * Sy)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool tCond = t > St1 && t < St2;
            double dx2 = 0.5 * Sx / Lx;
            double dy2 = 0.5 * Sy / Ly;
            double wf_c = 1 / (104857600 * M_PI * M_PI);
            double wf_d = (3969 * Sx * Sy) / 65536;
            double wf_x = (5040 * dx2 * M_PI + 2100 * Sx * std::sin(2 * M_PI * (-dx2 + xi) / Sx) -
                           600 * Sx * std::sin(4 * M_PI * (-dx2 + xi) / Sx) + 150 * Sx * std::sin(6 * M_PI * (-dx2 + xi) / Sx) -
                           25 * Sx * std::sin(8 * M_PI * (-dx2 + xi) / Sx) + 2 * Sx * std::sin(10 * M_PI * (-dx2 + xi) / Sx) -
                           2100 * Sx * std::sin(2 * M_PI * (dx2 + xi) / Sx) + 600 * Sx * std::sin(4 * M_PI * (dx2 + xi) / Sx) -
                           150 * Sx * std::sin(6 * M_PI * (dx2 + xi) / Sx) + 25 * Sx * std::sin(8 * M_PI * (dx2 + xi) / Sx) -
                           2 * Sx * std::sin(10 * M_PI * (dx2 + xi) / Sx));
            double wf_y = (5040 * dy2 * M_PI + 2100 * Sy * std::sin(2 * M_PI * (-dy2 + yj) / Sy) -
                           600 * Sy * std::sin(4 * M_PI * (-dy2 + yj) / Sy) + 150 * Sy * std::sin(6 * M_PI * (-dy2 + yj) / Sy) -
                           25 * Sy * std::sin(8 * M_PI * (-dy2 + yj) / Sy) + 2 * Sy * std::sin(10 * M_PI * (-dy2 + yj) / Sy) -
                           2100 * Sy * std::sin(2 * M_PI * (dy2 + yj) / Sy) + 600 * Sy * std::sin(4 * M_PI * (dy2 + yj) / Sy) -
                           150 * Sy * std::sin(6 * M_PI * (dy2 + yj) / Sy) + 25 * Sy * std::sin(8 * M_PI * (dy2 + yj) / Sy) -
                           2 * Sy * std::sin(10 * M_PI * (dy2 + yj) / Sy));

            double wf = wf_c * wf_x * wf_y / wf_d;
            Qh[j * Lx + i] = (tCond) ? w * wf : 0.0;
        }
    }
#elif SIMULATION_CASE == 2
    double Sx1 = (Sx - c) / 2.0 - b;
    double Sx2 = (Sx + c) / 2.0;
    double Sy1 = (Sy - a) / 2.0;
    double St1 = 50.0;
    double St2 = 60.0;
    double w = (p / (2 * a * b)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool tCond = t > St1 && t < St2;

            double dx2 = 0.5 * Sx / Lx;
            double dy2 = 0.5 * Sy / Ly;
            double A1 = std::max((std::min(Sx1 + b, xi + dx2) - std::max(Sx1, xi - dx2)), 0.0) * std::max((std::min(Sy1 + a, yj + dy2) - std::max(Sy1, yj - dy2)), 0.0);
            double A2 = std::max((std::min(Sx2 + b, xi + dx2) - std::max(Sx2, xi - dx2)), 0.0) * std::max((std::min(Sy1 + a, yj + dy2) - std::max(Sy1, yj - dy2)), 0.0);
            double Ac = 4.0 * dx2 * dy2;

            Qh[j * Lx + i] = (tCond) ? w * (A1 + A2) / Ac : 0.0;
        }
    }
#elif SIMULATION_CASE == 3
    double Sx1 = 0.2 * Sx;
    double Sx2 = 0.3 * Sx;
    double Sy1 = 0.2 * Sy;
    double Sy2 = 0.5 * Sy;
    double St1 = 50.0;
    double St2 = 60.0;
    double w = ((p / 2) / ((Sx2 - Sx1) * (Sy2 - Sy1))) / amp;
    MathCPU::Zero(Qh, Lx * Ly);

    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool tCond = t > St1 && t < St2;
            double dx2 = 0.5 * Sx / Lx;
            double dy2 = 0.5 * Sy / Ly;
            double A = std::max((std::min(Sx2, xi + dx2) - std::max(Sx1, xi - dx2)), 0.0) * std::max((std::min(Sy2, yj + dy2) - std::max(Sy1, yj - dy2)), 0.0);
            double Ac = 4.0 * dx2 * dy2;
            Qh[j * Lx + i] += (tCond) ? w * A / Ac : 0.0;
        }
    }

    w = (p / (1.6e-3 * Sx * Sy)) / amp;
    for (int j = 0; j < Ly; ++j)
    {
        for (int i = 0; i < Lx; ++i)
        {
            double xi = (i + 0.5) * Sx / Lx;
            double yj = (j + 0.5) * Sy / Ly;
            bool tCond = t > St1 && t < St2;
            double dx2 = 0.5 * Sx / Lx;
            double dy2 = 0.5 * Sy / Ly;
            double wf_c = 1.0 / (104857600.0 * M_PI * M_PI);
            double wf_d = (3969 * Sx * Sy) / 65536;
            double wf_x = (5040 * dx2 * M_PI + 2100 * Sx * std::sin(2 * M_PI * (-dx2 + xi) / Sx) -
                           600 * Sx * std::sin(4 * M_PI * (-dx2 + xi) / Sx) + 150 * Sx * std::sin(6 * M_PI * (-dx2 + xi) / Sx) -
                           25 * Sx * std::sin(8 * M_PI * (-dx2 + xi) / Sx) + 2 * Sx * std::sin(10 * M_PI * (-dx2 + xi) / Sx) -
                           2100 * Sx * std::sin(2 * M_PI * (dx2 + xi) / Sx) + 600 * Sx * std::sin(4 * M_PI * (dx2 + xi) / Sx) -
                           150 * Sx * std::sin(6 * M_PI * (dx2 + xi) / Sx) + 25 * Sx * std::sin(8 * M_PI * (dx2 + xi) / Sx) -
                           2 * Sx * std::sin(10 * M_PI * (dx2 + xi) / Sx));
            double wf_y = (5040 * dy2 * M_PI + 2100 * Sy * std::sin(2 * M_PI * (-dy2 + yj) / Sy) -
                           600 * Sy * std::sin(4 * M_PI * (-dy2 + yj) / Sy) + 150 * Sy * std::sin(6 * M_PI * (-dy2 + yj) / Sy) -
                           25 * Sy * std::sin(8 * M_PI * (-dy2 + yj) / Sy) + 2 * Sy * std::sin(10 * M_PI * (-dy2 + yj) / Sy) -
                           2100 * Sy * std::sin(2 * M_PI * (dy2 + yj) / Sy) + 600 * Sy * std::sin(4 * M_PI * (dy2 + yj) / Sy) -
                           150 * Sy * std::sin(6 * M_PI * (dy2 + yj) / Sy) + 25 * Sy * std::sin(8 * M_PI * (dy2 + yj) / Sy) -
                           2 * Sy * std::sin(10 * M_PI * (dy2 + yj) / Sy));

            double wf = wf_c * wf_x * wf_y / wf_d;
            Qh[j * Lx + i] += (tCond) ? w * wf : 0.0;
        }
    }
#endif
}

void Simulation(double *measures, double *Q_ref, double *Tc, int Lx, int Ly, int Lt, double Sx, double Sy, double Sz, double St, double amp, double h, double gamma, Type type)
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
    int Lu = 1 + 2 * (parms.Lx + parms.Ly);
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
    parms.gamma = gamma;

    Pointer<double> X, T, Q, aux;
    Pointer<double> U, T_amb, T_c;
    Pointer<double> Y, T_m;
    Pointer<double> XX, UX, XY;
    Pointer<double> Tr, Qr;
    X.alloc(L);
    T = X + Lsxy;
    Q = X;
    aux.alloc(L);

    U.alloc(Lu);
    T_amb = U;
    T_c = U + 1;

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
    double *T_ch = T_c.host();
    for (int i = 0; i < 2 * (Lx + Ly); ++i)
    {
        T_ch[i] = 300.0;
    }
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
        HC2D::GPU::EvolutionMatrix(parms, XX.dev(), UX.dev(), (Q.dev() - T.dev()), (T_c.dev() - T_amb.dev()));
        XX.copyDev2Host(L * L);
        HC2D::GPU::EvaluationMatrix(parms, XY.dev(), (Q.dev() - T.dev()));
        XY.copyDev2Host(L * Lsxy);
        cudaDeviceSynchronize();
    }
    else
    {
        HC2D::CPU::EvolutionMatrix(parms, XX.host(), UX.host(), (Q.host() - T.host()), (T_c.host() - T_amb.host()));
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
    for (int k = 0; k < 2 * (Lx + Ly) * Lt; ++k)
    {
        Tc[k] = 300.0;
    }
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

void ReadMeasurements(double *measures, double *Q_ref, double *T_c, int Lx, int Ly, int Lt, std::string case_input)
{
    std::ifstream inFile;
    int Lxy = Lx * Ly;
    for (int i = 0; i < Lt; i++)
    {
        inFile.open(case_input + "/Values" + std::to_string(i) + ".bin", std::ios::in | std::ios::binary);
        if (inFile.is_open())
        {
            inFile.read((char *)(measures + i * Lxy), sizeof(double) * Lxy);
        }
        else
        {
            throw std::runtime_error("Error reading file.");
        }
        inFile.close();
    }
    int Lp = 2 * Lx + 2 * Ly;
    for (int i = 0; i < Lt; i++)
    {
        inFile.open(case_input + "/Values" + std::to_string(i) + "_Tc.bin", std::ios::in | std::ios::binary);
        if (inFile.is_open())
        {
            inFile.read((char *)(T_c + i * Lp), sizeof(double) * Lp);
        }
        else
        {
            throw std::runtime_error("Error reading file.");
        }
        inFile.close();
    }
    MathCPU::Zero(Q_ref, Lx * Ly * Lt);
}

int main(int argc, char *argv[])
{
    Type type = Type::CPU;
    int Lx = LX_DEFAULT;
    int Ly = LY_DEFAULT;
    int Lt = LT_DEFAULT;
    double Sx = 0.0296;
    double Sy = 0.0296;
    double Sz = 0.0015;
    double St = 100.0;
    bool useMeasurements = false;
    std::string case_input;
    if (argc > 1)
    {
        type = (std::stoi(argv[1]) == 0) ? Type::CPU : Type::GPU;
    }
    if (argc > 5)
    {
        useMeasurements = std::stoi(argv[2]) != 0;
        case_input = argv[3];
        Lt = std::stoi(argv[4]);
        St = std::stod(argv[5]);
    }
    if (type == Type::GPU)
    {
        cudaDeviceReset();
        MathGPU::CreateHandles();
    }
    double amp = 5e3;
    double h = 0.0; // 11.0;
    double gamma = 0.0;

    std::ofstream outParms;
    outParms.open("data/kf/Parms.bin", std::ios::out | std::ios::binary);
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();

    Timer timer;
    HFE2D hfeKF;
    KF kf;

    hfeKF.SetParms(Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h);
    hfeKF.SetMemory(type);

    kf.SetModel(&hfeKF);
    kf.SetType(type);

    double *measures = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *measuresN = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *q_ref = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *T_c = (double *)malloc(sizeof(double) * 2 * (Lx + Ly) * Lt);
    if (useMeasurements)
    {
        ReadMeasurements(measures, q_ref, T_c, Lx, Ly, Lt, case_input);
    }
    else
    {
        Simulation(measures, q_ref, T_c, Lx, Ly, Lt, Sx, Sy, Sz, St, amp, h, gamma, type);
    }
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
        std::cout << "Loop: " << i + 1 << "/" << Lt << " at t = " << (i + 1) * St / Lt << "\n";

        kf.SetMeasure("Temperature", measuresN + Lx * Ly * i);
        kf.SetControl("Contour Temperature", T_c + 2 * (Lx + Ly) * i);

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

    kf.UnsetModel();
    hfeKF.UnsetMemory(type);

    if (type == Type::GPU)
    {
        MathGPU::DestroyHandles();
        cudaDeviceReset();
    }
    return 0;
}