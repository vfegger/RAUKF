#include "./src/hfe/include/hfe.hpp"
#include "./src/hfe/include/hc.hpp"
#include "./src/filter/include/raukf.hpp"
#include "./src/filter/include/kf.hpp"

#include <fstream>
#include <random>

#define RAUKF_USAGE 1
#define KF_USAGE 1
#define KF_AEM_USAGE 1
#define USE_RADIATION 1
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

void Simulation(double *measures, double *Q_ref, int Lx, int Ly, int Lz, int Lt, double Sx, double Sy, double Sz, double St, double amp, double T_amb, double T_ref, double eps, Type type)
{
    double *workspace;
    HC::HCParms parms;
    int Lsx = 4 * Lx;
    int Lsy = 4 * Ly;
    int Lsz = 4 * Lz;
    int n = 100;
    int Lst = n * Lt;
    parms.Lx = Lsx;
    parms.Ly = Lsy;
    parms.Lz = Lsz;
    parms.Lt = Lst;
    parms.Sx = Sx;
    parms.Sy = Sy;
    parms.Sz = Sz;
    parms.St = St;
    parms.dx = Sx / Lsx;
    parms.dy = Sy / Lsy;
    parms.dz = Sz / Lsz;
    parms.dt = St / Lst;
    parms.amp = amp;
    parms.T_amb = T_amb;
    parms.T_ref = T_ref;
    parms.eps = eps;

    Pointer<double> T, Q;
    Pointer<double> Tr, Qr;
    T.alloc(Lsx * Lsy * Lsz);
    Q.alloc(Lsx * Lsy);
    Tr.alloc(Lt * Lx * Ly);
    Qr.alloc(Lt * Lx * Ly);
    Math::Zero(T, Lsx * Lsy * Lsz, Type::CPU);
    Math::Zero(Q, Lsx * Lsy, Type::CPU);

    double *Th = T.host();
    for (int i = 0; i < Lsx * Lsy * Lsz; ++i)
    {
        Th[i] = 300.0;
    }
    if (type == Type::GPU)
    {
        T.copyHost2Dev(Lsx * Lsy * Lsz);
        Q.copyHost2Dev(Lsx * Lsy);
        HC::GPU::AllocWorkspaceRKF45(workspace, parms);
        cudaDeviceSynchronize();
    }
    else
    {
        HC::CPU::AllocWorkspaceRKF45(workspace, parms);
    }
    for (int k = 0; k < Lst; ++k)
    {
        // Heat Flux Definition
        double *Qh = Q.host();
        for (int j = 0; j < Lsy; ++j)
        {
            for (int i = 0; i < Lsx; ++i)
            {
                bool xCond = (i + 0.5) * Sx / Lsx > 0.3 * Sx && (i + 0.5) * Sx / Lsx < 0.7 * Sx;
                bool yCond = (j + 0.5) * Sy / Lsy > 0.3 * Sy && (j + 0.5) * Sy / Lsy < 0.7 * Sy;
                bool tCond = k * St / Lst > 1.0 && k * St / Lst < 5.0;
                Qh[j * Lsx + i] = (xCond && yCond && tCond) ? 100.0 : 0.0;
            }
        }
        if (type == Type::GPU)
        {
            Q.copyHost2Dev(Lsx * Lsy);
            HC::GPU::RK4(T.dev(), Q.dev(), workspace, parms);
            cudaDeviceSynchronize();
        }
        else
        {
            HC::CPU::RK4(T.host(), Q.host(), workspace, parms);
        }
        if (k % n == n - 1)
        {
            Interpolation::Rescale(T, Lsx, Lsy, Tr + (k / n) * Lx * Ly, Lx, Ly, Sx, Sy, type);
            Interpolation::Rescale(Q, Lsx, Lsy, Qr + (k / n) * Lx * Ly, Lx, Ly, Sx, Sy, type);
            cudaDeviceSynchronize();
        }
    }

    if (type == Type::GPU)
    {
        Tr.copyDev2Host(Lt * Lx * Ly);
        Qr.copyDev2Host(Lt * Lx * Ly);
        cudaDeviceSynchronize();
    }

    MathCPU::Copy(measures, Tr.host(), Lx * Ly * Lt);
    MathCPU::Copy(Q_ref, Qr.host(), Lx * Ly * Lt);
    if (type == Type::GPU)
    {
        HC::GPU::FreeWorkspaceRKF45(workspace);
    }
    else
    {
        HC::CPU::FreeWorkspaceRKF45(workspace);
    }
    Qr.free();
    Tr.free();
    Q.free();
    T.free();
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
    int Lx = 24;
    int Ly = 24;
    int Lz = 6;
    int Lt = 500;
    double Sx = 0.12;
    double Sy = 0.12;
    double Sz = 0.003;
    double St = 10.0;
    double amp = 5e4;
#if USE_RADIATION == 1
    double epsC = 1.0;
#else
    double epsC = 0.0;
#endif
    double epsR = 0.0;

    std::ofstream outParms;
    int temp;
#if RAUKF_USAGE == 1
    outParms.open("data/raukf/Parms.bin", std::ios::out | std::ios::binary);
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&Lz), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();
#endif
#if KF_USAGE == 1
    outParms.open("data/kf/Parms.bin", std::ios::out | std::ios::binary);
    temp = 1;
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&temp), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();
#endif
#if KF_AEM_USAGE == 1
    outParms.open("data/kfaem/Parms.bin", std::ios::out | std::ios::binary);
    temp = 1;
    outParms.write((char *)(&Lx), sizeof(int));
    outParms.write((char *)(&Ly), sizeof(int));
    outParms.write((char *)(&temp), sizeof(int));
    outParms.write((char *)(&Lt), sizeof(int));
    outParms.close();
#endif

    Timer timer;
#if RAUKF_USAGE == 1
    HFE hfe;
#endif

#if KF_USAGE == 1
    HFE_RM hfeKF;
#endif

#if KF_AEM_USAGE == 1
    HFE hfeC;
    HFE_RM hfeR;
    HFE_AEM hfeAEM;
#endif

#if RAUKF_USAGE == 1
    RAUKF raukf;
#endif
#if KF_USAGE == 1
    KF kf;
#endif
#if KF_AEM_USAGE == 1
    KF kfAEM;
#endif

#if RAUKF_USAGE == 1
    hfe.SetParms(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp, 300.0, 600.0, epsC);
    hfe.SetMemory(type);
#endif
#if KF_USAGE == 1
    hfeKF.SetParms(Lx, Ly, Lt, Sx, Sy, Sz, St, amp, 300.0, 600.0, epsR);
    hfeKF.SetMemory(type);
#endif
#if KF_AEM_USAGE == 1
    hfeC.SetParms(Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp, 300.0, 600.0, epsC);
    hfeC.SetMemory(type);
    hfeR.SetParms(Lx, Ly, Lt, Sx, Sy, Sz, St, amp, 300.0, 600.0, epsR);
    hfeR.SetMemory(type);
    hfeAEM.SetModel(&hfeR, &hfeC);
    hfeAEM.SetMemory(type);
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

#if KF_AEM_USAGE == 1
    kfAEM.SetModel(&hfeAEM);
    kfAEM.SetType(type);
#endif

    double *measures = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *measuresN = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    double *q_ref = (double *)malloc(sizeof(double) * Lx * Ly * Lt);
    Simulation(measures, q_ref, Lx, Ly, Lz, Lt, Sx, Sy, Sz, St, amp, 300.0, 600.0, epsC, type);
    double *resultT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarT = (double *)malloc(sizeof(double) * Lx * Ly * Lz);
    double *resultCovarQ = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultTm = (double *)malloc(sizeof(double) * Lx * Ly);
    double *resultCovarTm = (double *)malloc(sizeof(double) * Lx * Ly);

    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 5.0);
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
            outFile.write((char *)resultT, sizeof(double) * Lx * Ly * Lz);
            outFile.write((char *)resultCovarT, sizeof(double) * Lx * Ly * Lz);
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

#if KF_AEM_USAGE == 1
        kfAEM.SetMeasure("Temperature", measuresN + Lx * Ly * i);

        kfAEM.Iterate(timer);
        kfAEM.GetState("Temperature", resultT);
        kfAEM.GetState("Heat Flux", resultQ);
        kfAEM.GetStateCovariance("Temperature", resultCovarT);
        kfAEM.GetStateCovariance("Heat Flux", resultCovarQ);
        kfAEM.GetMeasure("Temperature", resultTm);
        kfAEM.GetMeasureCovariance("Temperature", resultCovarTm);

        outFile.open("data/kfaem/Values" + std::to_string(i) + ".bin", std::ios::out | std::ios::binary);
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

        outFile.open("data/kfaem/ready/" + std::to_string(i), std::ios::out | std::ios::binary);
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

#if KF_AEM_USAGE == 1
    kfAEM.UnsetModel();
#endif
#if KF_USAGE == 1
    kf.UnsetModel();
#endif
#if RAUKF_USAGE == 1
    raukf.UnsetWeight();
    raukf.UnsetModel();
#endif
#if KF_AEM_USAGE == 1
    hfeAEM.UnsetMemory(type);
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