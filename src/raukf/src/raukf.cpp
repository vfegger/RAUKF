#include "../include/raukf.hpp"

RAUKF::RAUKF() : pmodel(NULL), alpha(0.0), beta(0.0), kappa(0.0), lambda(0.0), type(Type::CPU) {}

void RAUKF::SetModel(Model *pmodel)
{
    this->pmodel = pmodel;
}

void RAUKF::SetState(Data *pstate)
{
    this->pstate = pstate;
}

void RAUKF::SetType(Type type)
{
    this->type = type;
}

void RAUKF::Iterate(Timer &timer)
{
    // Preparation of timer class
    timer.Reset();
    timer.Start();

    // Initialization of variables
    Pointer<double> x = this->pstate->GetStatePointer();
    Pointer<double> Pxx = this->pstate->GetStateCovariancePointer();
    Pointer<double> xs = this->pstate->GetInstances();
    Pointer<double> Q;

    Pointer<double> y = this->pmeasure->GetMeasurePointer();
    Pointer<double> Pyy = this->pmeasure->GetMeasureCovariancePointer();
    Pointer<double> ys = this->pmeasure->GetInstances();
    Pointer<double> z = this->pmeasure->GetMeasurePointer();
    Pointer<double> R;

    Pointer<double> PxyT;
    Pointer<double> KT;

    int Lx = this->pstate->GetStateLength();
    int Ls = this->pstate->GetSigmaLength();
    int Ly = this->pmeasure->GetMeasureLength();

    PxyT.alloc(Lx * Ly);
    KT.alloc(Lx * Ly);

    std::cout << "State length: " << Lx << "; Observation length: " << Ly << "; Sigma points: " << Ls << "\n";

    // Generation of sigma points through the use of cholesky decomposition
    std::cout << "Calculate Cholesky Decomposition\n";
    Pointer<double> cd;
    cd.alloc(Lx * Lx);
    Math::Mul(Pxx, Lx + lambda, Lx * Lx, type);
    Math::CholeskyDecomposition(cd, Pxx, Lx, type);

    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    Math::Iterate(Math::Copy, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Add, xs, cd, Lx, Lx, Lx, Lx, Lx, 0, type);
    Math::Iterate(Math::Sub, xs, cd, Lx, Lx, Lx, Lx, Lx * (Lx + 1), 0, type);

    cd.free();
    // Evolve and Measure each state given by each sigma point
    std::cout << "Evolution Step\n";
    for (int i = 0; i < Ls; ++i)
    {
        pmodel->Evolve(pstate, type);
    }
    std::cout << "Evaluation Step\n";
    for (int i = 0; i < Ls; ++i)
    {
        pmodel->Evaluate(pmeasure, pstate, type);
    }

    // Calculate new mean and covariance for the state and measure
    std::cout << "Mean\n";
    Math::Mean(x, xs, Lx, Ls, type);
    Math::Mean(y, ys, Ly, Ls, type);

    std::cout << "Covariance\n";
    Math::Iterate(Math::Sub, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);

    Math::MatMulNT(0.0, Pxx, 1.0, xs, xs, Lx, Ls, Lx, type);
    Math::MatMulNT(0.0, Pyy, 1.0, ys, ys, Ly, Ls, Ly, type);
    Math::MatMulNT(0.0, PxyT, 1.0, ys, xs, Ly, Ls, Lx, type);
    Math::Add(Pxx, Q, Lx * Lx, type);
    Math::Add(Pyy, R, Ly * Ly, type);

    // Kalman gain calculation by solving: Pyy * (K^T) = Pxy^T
    Math::CholeskySolver(KT, Pyy, PxyT, Ly, Ly, Lx, type);

    // State Update
    std::cout << "State Update\n";
    Math::Sub(z, y, Ly, type);
    Math::MatMulTN(1.0, x, 1.0, KT, z, Lx, Ly, 1, type);

    std::cout << "State Covariance Update\n";
    Math::MatMulTN(0.0, PxyT, 1.0, KT, Pyy, Lx, Ly, Ly, type);
    Math::MatMulNN(1.0, Pxx, 1.0, PxyT, KT, Lx, Ly, Ly, type);
}
