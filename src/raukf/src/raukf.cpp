#include "../include/raukf.hpp"

#include <iomanip>
#include <fstream>

RAUKF::RAUKF() : pstatistics(NULL), pmodel(NULL), pstate(NULL), pmeasure(NULL), alpha(0.0), beta(0.0), kappa(0.0), type(Type::CPU) {
    pstatistics = new Statistics();
}

RAUKF::~RAUKF() {
    delete pstatistics;
}

void RAUKF::SetParameters(double alpha, double beta, double kappa)
{
    this->alpha = alpha;
    this->beta = beta;
    this->kappa = kappa;
}

void RAUKF::SetModel(Model *pmodel)
{
    if (this->pmodel == NULL)
    {
        this->pmodel = pmodel;
        this->pstate = pmodel->GenerateData();
        this->pmeasure = pmodel->GenerateMeasure();
        pstate->SetInstances();
        pmeasure->SetInstances(this->pstate->GetStateLength());
    }
    else
    {
        std::cout << "Error: Model is not NULL";
    }
}

void RAUKF::UnsetModel()
{
    if (this->pmodel != NULL)
    {
        pmeasure->UnsetInstances();
        pstate->UnsetInstances();
        delete this->pstate;
        delete this->pmeasure;
        this->pmodel = NULL;
    }
    else
    {
        std::cout << "Error: Model is NULL";
    }
}

void RAUKF::SetType(Type type)
{
    this->type = type;
}

void RAUKF::SetWeight()
{
    int Lx = this->pstate->GetStateLength();
    int Ls = this->pstate->GetSigmaLength();
    this->lambda = alpha * alpha * (Lx + kappa) - Lx;
    wm.alloc(Ls);
    wc.alloc(Ls);
    double *pwmhost = wm.host();
    double *pwchost = wc.host();
    pwmhost[0] = lambda / (Lx + lambda);
    pwchost[0] = lambda / (Lx + lambda) + 1.0 - alpha * alpha + beta;
    for (int i = 1; i < Ls; ++i)
    {
        pwmhost[i] = pwchost[i] = 0.5 / (Lx + lambda);
    }
    wm.copyHost2Dev(Ls);
    wc.copyHost2Dev(Ls);
    cudaDeviceSynchronize();
}

void RAUKF::UnsetWeight()
{
    wm.free();
    wc.free();
}

void RAUKF::SetMeasure(std::string name, double *data)
{
    this->pmeasure->SetMeasureData(name, data);
}

void RAUKF::GetState(std::string name, double *data)
{
    this->pstate->GetStateData(name, data);
}

void RAUKF::GetStateCovariance(std::string name, double *data)
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

void RAUKF::Iterate(Timer &timer)
{
    timer.Record(type);

    // Initialization of variables
    Pointer<double> x = this->pstate->GetStatePointer();
    Pointer<double> Pxx = this->pstate->GetStateCovariancePointer();
    Pointer<double> xs = this->pstate->GetInstances();
    Pointer<double> Q = this->pstate->GetStateNoisePointer();

    Pointer<double> y = this->pmeasure->GetMeasurePointer();
    Pointer<double> Pyy = this->pmeasure->GetMeasureCovariancePointer();
    Pointer<double> ys = this->pmeasure->GetInstances();
    Pointer<double> ym = this->pmeasure->GetMeasureData();
    Pointer<double> R = this->pmeasure->GetMeasureNoisePointer();

    Pointer<double> PxyT;
    Pointer<double> KT;
    Pointer<double> cd;

    int Lx = this->pstate->GetStateLength();
    int Ls = this->pstate->GetSigmaLength();
    int Ly = this->pmeasure->GetMeasureLength();

    PxyT.alloc(Lx * Ly);
    KT.alloc(Lx * Ly);
    cd.alloc(Lx * Lx);

    std::cout << "State length: " << Lx << "; Observation length: " << Ly << "; Sigma points: " << Ls << "\n";
    timer.Record(type);

    // Generation of sigma points through the use of cholesky decomposition
    std::cout << "Calculate Cholesky Decomposition\n";
    Math::Zero(cd, Lx * Lx, type);
    Math::Mul(Pxx, Lx + lambda, Lx * Lx, type);
    Math::CholeskyDecomposition(cd, Pxx, Lx, type);
    timer.Record(type);

    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    Math::Iterate(Math::Copy, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Add, xs, cd, Lx, Lx, Lx, Lx, Lx, 0, type);
    Math::Iterate(Math::Sub, xs, cd, Lx, Lx, Lx, Lx, Lx * (Lx + 1), 0, type);

    timer.Record(type);
    // Evolve and Measure each state given by each sigma point
    std::cout << "Evolution Step\n";
    pmodel->Evolve(pstate, type);
    timer.Record(type);

    std::cout << "Evaluation Step\n";
    pmodel->Evaluate(pmeasure, pstate, type);
    timer.Record(type);

    // Calculate new mean and covariance for the state and measure
    std::cout << "Mean\n";
    Math::Mean(x, xs, wm, Lx, Ls, type);
    Math::Mean(y, ys, wm, Ly, Ls, type);
    timer.Record(type);

    std::cout << "Covariance\n";
    Math::Iterate(Math::Sub, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);

    Math::MatMulNWT(0.0, Pxx, 1.0, xs, xs, wc, Lx, Ls, Lx, type);
    Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);
    Math::MatMulNWT(0.0, PxyT, 1.0, ys, xs, wc, Ly, Ls, Lx, type);
    Math::Add(Pxx, Q, Lx * Lx, type);
    Math::Add(Pyy, R, Ly * Ly, type);
    timer.Record(type);

    // Kalman gain calculation by solving: Pyy * (K^T) = Pxy^T
    std::cout << "Kalman Gain\n";
    Math::CholeskySolver(KT, Pyy, PxyT, Ly, Ly, Lx, type);
    timer.Record(type);

    // State Update
    std::cout << "State Update\n";
    Math::Sub(ym, y, Ly, type);
    Math::MatMulTN(1.0, x, 1.0, KT, ym, Lx, Ly, 1, type);
    timer.Record(type);

    std::cout << "State Covariance Update\n";
    Math::MatMulTN(0.0, PxyT, 1.0, KT, Pyy, Lx, Ly, Ly, type);
    Math::MatMulNN(1.0, Pxx, -1.0, PxyT, KT, Lx, Ly, Ly, type);
    timer.Record(type);

    cd.free();
    KT.free();
    PxyT.free();
    if (type == Type::GPU)
    {
        x.copyDev2Host(Lx);
        Pxx.copyDev2Host(Lx * Lx);
        cudaDeviceSynchronize();
    }
}
