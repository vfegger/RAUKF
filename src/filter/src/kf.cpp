#include "../include/kf.hpp"

#include <iomanip>
#include <fstream>

KF::KF() : pstatistics(NULL), pmodel(NULL), pstate(NULL), pmeasure(NULL), type(Type::CPU)
{
    pstatistics = new Statistics();
}

KF::~KF()
{
    delete pstatistics;
}

void KF::SetModel(LinearModel *pmodel)
{
    if (this->pmodel == NULL)
    {
        this->pmodel = pmodel;
        this->pstate = pmodel->GenerateData();
        this->pmeasure = pmodel->GenerateMeasure();
    }
    else
    {
        std::cout << "Error: Model is not NULL";
    }
}

void KF::UnsetModel()
{
    if (this->pmodel != NULL)
    {
        delete this->pstate;
        delete this->pmeasure;
        this->pmodel = NULL;
    }
    else
    {
        std::cout << "Error: Model is NULL";
    }
}

void KF::SetType(Type type)
{
    this->type = type;
}

void KF::SetMeasure(std::string name, double *data)
{
    this->pmeasure->SetMeasureData(name, data);
}

void KF::GetState(std::string name, double *data)
{
    this->pstate->GetStateData(name, data);
}

void KF::GetStateCovariance(std::string name, double *data)
{
    this->pstate->GetStateCovarianceData(name, data);
}

void PrintMatrix(std::string name, double *mat, int lengthI, int lengthJ)
{
    std::ofstream fp;
    fp.open(name + ".csv");
    for (int i = 0; i < lengthI; ++i)
    {
        for (int j = 0; j < lengthJ; ++j)
        {
            fp << mat[j * lengthI + i];
            if (j < lengthJ)
            {
                fp << ",";
            }
        }
        fp << "\n";
    }
    fp.close();
}

void PrintMatrix(std::string name, Pointer<double> mat, int lengthI, int lengthJ, Type type)
{
    if (type == Type::GPU)
    {
        cudaDeviceSynchronize();
        mat.copyDev2Host(lengthI * lengthJ);
    }
    std::cout << name << ":\n";
    double *p = mat.host();
    std::cout.precision(2);
    std::cout << std::fixed;
    for (int i = 0; i < lengthI; ++i)
    {
        for (int j = 0; j < lengthJ; ++j)
        {
            std::cout << p[j * lengthI + i];
            if (j < lengthJ)
            {
                std::cout << ",";
            }
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

void KF::Iterate(Timer &timer)
{
    timer.Record(type);

    // Initialization of variables
    Pointer<double> x = this->pstate->GetStatePointer();
    Pointer<double> Pxx = this->pstate->GetStateCovariancePointer();
    Pointer<double> Q = this->pstate->GetStateNoisePointer();

    Pointer<double> y = this->pmeasure->GetMeasurePointer();
    Pointer<double> Pyy = this->pmeasure->GetMeasureCovariancePointer();
    Pointer<double> ym = this->pmeasure->GetMeasureData();
    Pointer<double> R = this->pmeasure->GetMeasureNoisePointer();

    Pointer<double> KT;
    Pointer<double> F;
    Pointer<double> H;

    Pointer<double> v_0;
    Pointer<double> v_1;

    Pointer<double> workspaceLx2;
    Pointer<double> workspaceLxLy;

    int Lx = this->pstate->GetStateLength();
    int Ly = this->pmeasure->GetMeasureLength();

    F.alloc(Lx * Lx);
    H.alloc(Ly * Lx);
    KT.alloc(Lx * Ly);
    v_0.alloc(Ly);
    v_1.alloc(Ly);
    workspaceLx2.alloc(Lx * Lx);
    workspaceLxLy.alloc(Lx * Ly);

    std::cout << "Iteration:\n\tState length: " << Lx << "; Observation length: " << Ly << "\n";
    timer.Record(type);

    pmodel->Evolution(F, pstate, type);
    timer.Record(type);

    pmodel->Evaluation(H, pmeasure, pstate, type);
    timer.Record(type);

    Math::MatMulNN(0.0, x, 1.0, F, x, Lx, Lx, 1, type);
    Math::MatMulNN(0.0, Pxx, 1.0, F, Pxx, Lx, Lx, Lx, type);
    Math::MatMulNT(0.0, Pxx, 1.0, Pxx, F, Lx, Lx, Lx, type);
    Math::Add(Pxx, Q, Lx * Lx, type);
    timer.Record(type);

    Math::MatMulNN(0.0, y, 1.0, H, x, Ly, Lx, 1, type);
    Math::MatMulNN(0.0, Pyy, 1.0, H, Pxx, Ly, Lx, Lx, type);
    Math::MatMulNT(0.0, Pyy, 1.0, Pxx, H, Ly, Lx, Ly, type);
    Math::Add(Pyy, R, Ly * Ly, type);
    timer.Record(type);

    Math::MatMulNT(0.0, workspaceLxLy, 1.0, H, Pxx, Ly, Lx, Lx, type);
    Math::CholeskySolver(KT, Pyy, workspaceLxLy, Ly, Ly, Lx, type);
    timer.Record(type);

    Math::Sub(v_0, ym, y, Ly, type);
    Math::MatMulTN(1.0, x, 1.0, KT, v_0, Lx, Ly, 1, type);
    Math::MatMulTN(0.0, workspaceLx2, 1.0, KT, H, Lx, Ly, Lx, type);
    Math::MatMulNN(1.0, Pxx, -1.0, workspaceLx2, Pxx, Lx, Lx, Lx, type);

    Math::MatMulNN(0.0, y, 1.0, H, x, Ly, Lx, 1, type);
    Math::Sub(v_1, ym, y, Ly, type);
    timer.Record(type);

    workspaceLxLy.free();
    workspaceLx2.free();
    v_1.free();
    v_0.free();
    KT.free();
    H.free();
    F.free();

    if (type == Type::GPU)
    {
        x.copyDev2Host(Lx);
        Pxx.copyDev2Host(Lx * Lx);
        v_0.copyDev2Host(Ly);
        v_1.copyDev2Host(Ly);
        cudaDeviceSynchronize();
    }
}