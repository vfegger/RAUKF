#include "../include/raukf.hpp"

#include <iomanip>
#include <fstream>

RAUKF::RAUKF() : pstatistics(NULL), pmodel(NULL), pstate(NULL), pmeasure(NULL), alpha(0.0), beta(0.0), kappa(0.0), type(Type::CPU)
{
    pstatistics = new Statistics();
}

RAUKF::~RAUKF()
{
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

void RAUKF::UnscentedTransformation(Pointer<double> xs_o, Pointer<double> x_i, Pointer<double> P_i, int Lx, int Ls, Pointer<double> workspace, Type type)
{
    std::cout << "Calculate Cholesky Decomposition\n";
    Math::Zero(workspace, Lx * Lx, type);
    Math::CholeskyDecomposition(workspace, P_i, Lx, type);
    Math::Mul(workspace, sqrt(Lx + lambda), Lx * Lx, type);

    std::cout << "Generate Sigma Points based on Cholesky Decompostion\n";
    Math::Iterate(Math::Copy, xs_o, x_i, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Add, xs_o, workspace, Lx, Lx, Lx, Lx, Lx, 0, type);
    Math::Iterate(Math::Sub, xs_o, workspace, Lx, Lx, Lx, Lx, Lx * (Lx + 1), 0, type);
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
    Pointer<double> workspace;

    Pointer<double> mu;
    Pointer<double> aux;

    int Lx = this->pstate->GetStateLength();
    int Ls = this->pstate->GetSigmaLength();
    int Ly = this->pmeasure->GetMeasureLength();

    PxyT.alloc(Lx * Ly);
    KT.alloc(Lx * Ly);
    workspace.alloc(Lx * Lx);
    mu.alloc(Ly);
    aux.alloc(Ly * Ly);

    std::cout << "State length: " << Lx << "; Observation length: " << Ly << "; Sigma points: " << Ls << "\n";
    timer.Record(type);

    // Unscented Transformation of the state
    UnscentedTransformation(xs, x, Pxx, Lx, Ls, workspace, type);

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
    Math::Sub(mu, ym, y, Ly, type);
    Math::MatMulTN(1.0, x, 1.0, KT, mu, Lx, Ly, 1, type);
    timer.Record(type);

    std::cout << "State Covariance Update\n";
    Math::MatMulTN(0.0, PxyT, 1.0, KT, Pyy, Lx, Ly, Ly, type);
    Math::MatMulNN(1.0, Pxx, -1.0, PxyT, KT, Lx, Ly, Ly, type);
    timer.Record(type);

    // Calculation of Correction Factor
    Math::CholeskySolver(aux, Pyy, mu, Ly, Ly, 1, type);
    double phi = Math::Dot(mu, aux, Ly, type);

    double chi2 = pstatistics->GetChi2(0.95, Ly);
    if (phi > chi2)
    {
        std::cout << "Chi-Squared Criterion Violated:\n";
        std::cout << "\tUpdate Noise Covariances\n";
        // Update Noise Matrix Q
        double a = 5.0;
        double lambdaQ0 = 0.2;
        double lambdaQ = std::max<double>(lambdaQ0, (phi - a * chi2) / phi);
        Math::MatMulTN(0.0, aux, 1.0, mu, KT, 1, Ly, Lx, type);
        Math::MatMulTN(1.0 - lambdaQ, Q, lambdaQ, aux, aux, Lx, 1, Lx, type);

        // Re-sample sigma points with the new state
        UnscentedTransformation(xs, x, Pxx, Lx, Ls, workspace, type);
        pmodel->Evaluate(pmeasure, pstate, type);

        Math::Mean(y, ys, wm, Ly, Ls, type);
        Math::Sub(mu, ym, xs, Ly, type);

        Math::Iterate(Math::Sub, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
        Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);

        Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);

        // Update Noise Matrix R
        double b = 5.0;
        double lambdaR0 = 0.2;
        double lambdaR = std::max<double>(lambdaR0, (phi - b * chi2) / phi);
        Math::MatMulTN(1.0 - lambdaR, R, lambdaR, mu, mu, Ly, 1, Ly, type);
        Math::LRPO(R, Pyy, lambdaR, Ly * Ly, type);

        // Update Matrices for State and Covariance Update
        Math::MatMulNWT(0.0, Pxx, 1.0, xs, xs, wc, Lx, Ls, Lx, type);
        // Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);
        Math::MatMulNWT(0.0, PxyT, 1.0, ys, xs, wc, Ly, Ls, Lx, type);
        Math::Add(Pxx, Q, Lx * Lx, type);
        Math::Add(Pyy, R, Ly * Ly, type);

        Math::CholeskySolver(KT, Pyy, PxyT, Ly, Ly, Lx, type);

        std::cout << "\tState Update\n";
        Math::Sub(mu, ym, y, Ly, type);
        Math::MatMulTN(1.0, x, 1.0, KT, mu, Lx, Ly, 1, type);
        timer.Record(type);

        std::cout << "\tState Covariance Update\n";
        Math::MatMulTN(0.0, PxyT, 1.0, KT, Pyy, Lx, Ly, Ly, type);
        Math::MatMulNN(1.0, Pxx, -1.0, PxyT, KT, Lx, Ly, Ly, type);
        timer.Record(type);
    }
    aux.free();
    mu.free();

    workspace.free();
    KT.free();
    PxyT.free();
    if (type == Type::GPU)
    {
        x.copyDev2Host(Lx);
        Pxx.copyDev2Host(Lx * Lx);
        cudaDeviceSynchronize();
    }
}
