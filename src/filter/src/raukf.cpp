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

void RAUKF::SetControl(std::string name, double *data)
{
    this->pcontrol->SetControlData(name, data);
}

void RAUKF::GetControl(std::string name, double *data)
{
    this->pcontrol->GetControlData(name, data);
}

void RAUKF::SetMeasure(std::string name, double *data)
{
    this->pmeasure->SetMeasureData(name, data);
}

void RAUKF::GetMeasure(std::string name, double *data)
{
    this->pmeasure->GetMeasureData(name, data);
}

void RAUKF::GetMeasureCovariance(std::string name, double *data)
{
    this->pmeasure->GetMeasureCovarianceData(name, data);
}

void RAUKF::GetState(std::string name, double *data)
{
    this->pstate->GetStateData(name, data);
}

void RAUKF::GetStateCovariance(std::string name, double *data)
{
    this->pstate->GetStateCovarianceData(name, data);
}

void RAUKF::PrintMatrix(std::string name, double *mat, int lengthI, int lengthJ)
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

void RAUKF::PrintMatrix(std::string name, Pointer<double> mat, int lengthI, int lengthJ, Type type)
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

void RAUKF::UnscentedTransformation(Pointer<double> xs_o, Pointer<double> x_i, Pointer<double> P_i, int Lx, int Ls, Pointer<double> workspace, Type type)
{
    Math::Zero(workspace, Lx * Lx, type);
    Math::CholeskyDecomposition(workspace, P_i, Lx, type);
    Math::Mul(workspace, sqrt(Lx + lambda), Lx * Lx, type);
    Math::Iterate(Math::Copy, xs_o, x_i, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Add, xs_o, workspace, Lx, Lx, Lx, Lx, Lx, 0, type);
    Math::Iterate(Math::Sub, xs_o, workspace, Lx, Lx, Lx, Lx, Lx * (Lx + 1), 0, type);
}

void RAUKF::AdaptNoise(Pointer<double> x, Pointer<double> Pxx, Pointer<double> Q, Pointer<double> xs, Pointer<double> y, Pointer<double> Pyy, Pointer<double> R, Pointer<double> ys, Pointer<double> ym, Pointer<double> v_0, Pointer<double> v_1, Pointer<double> PxyT, Pointer<double> KT, int Lx, int Ls, int Ly, Pointer<double> workspaceLx, Pointer<double> workspaceLy, Pointer<double> workspaceLx2, Pointer<double> workspaceLxLy)
{
    // Calculation of Correction Factor
    pmodel->Evaluate(pmeasure, pstate, ExecutionType::State, type);
    Math::Sub(v_1, ym, y, Ly, type);
    Math::CholeskySolver(workspaceLy, Pyy, v_1, Ly, Ly, 1, type);
    double phi = Math::Dot(v_1, workspaceLy, Ly, type);
    double chi2 = pstatistics->GetChi2(0.005, Ly);
    std::cout << "\tChi-Squared Criterion: " << chi2 << "; Fault Detection Rule: " << phi << "\n";
    if (phi > chi2)
    {
        std::cout << "\t\tChi-Squared Criterion Violated: " << chi2 << " < " << phi << "\n";

        // Update Noise Matrix Q
        double a = 5.0;
        double lambdaQ0 = 0.01;
        double lambdaQ = std::max<double>(lambdaQ0, (phi - a * chi2) / phi);
        Math::MatMulTN(0.0, workspaceLx, 1.0, v_0, KT, 1, Ly, Lx, type);
        Math::MatMulNT(1.0 - lambdaQ, Q, lambdaQ, workspaceLx, workspaceLx, Lx, 1, Lx, type);

        // Update Noise Matrix R
        double b = 5.0;
        double lambdaR0 = 0.01;
        double lambdaR = std::max<double>(lambdaR0, (phi - b * chi2) / phi);

        UnscentedTransformation(xs, x, Pxx, Lx, Ls, workspaceLx2, type);
        pmodel->Evaluate(pmeasure, pstate, ExecutionType::Instance, type);

        Math::MatMulNT(1.0 - lambdaR, R, lambdaR, v_1, v_1, Ly, 1, Ly, type);

        Math::Mean(y, ys, wm, Ly, Ls, type);
        Math::Iterate(Math::Sub, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
        Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);
        Math::MatMulNWT(0.0, Pxx, 1.0, xs, xs, wc, Lx, Ls, Lx, type);
        Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);
        Math::MatMulNWT(0.0, PxyT, 1.0, ys, xs, wc, Ly, Ls, Lx, type);
        Math::Diag(workspaceLy, Pyy, Ly, type);

        Math::LRPO(R, Pyy, lambdaR, Ly * Ly, type);

        // Update Matrices for State and Covariance Update
        Math::Add(Pxx, Q, Lx * Lx, type);
        Math::Add(Pyy, R, Ly * Ly, type);

        Math::CholeskySolver(KT, Pyy, PxyT, Ly, Ly, Lx, type);

        Math::Sub(workspaceLy, ym, y, Ly, type);
        Math::MatMulTN(1.0, x, 1.0, KT, workspaceLy, Lx, Ly, 1, type);

        Math::MatMulTN(0.0, workspaceLxLy, 1.0, KT, Pyy, Lx, Ly, Ly, type);
        Math::MatMulNN(1.0, Pxx, -1.0, workspaceLxLy, KT, Lx, Ly, Lx, type);
    }
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

    Pointer<double> u = this->pcontrol->GetControlPointer();

    Pointer<double> PxyT;
    Pointer<double> KT;
    Pointer<double> workspaceLx;
    Pointer<double> workspaceLy;
    Pointer<double> workspaceLx2;
    Pointer<double> workspaceLxLy;

    Pointer<double> v_0;
    Pointer<double> v_1;

    int Lx = this->pstate->GetStateLength();
    int Ls = this->pstate->GetSigmaLength();
    int Ly = this->pmeasure->GetMeasureLength();
    int Lu = this->pcontrol->GetControlLength();

    PxyT.alloc(Lx * Ly);
    KT.alloc(Lx * Ly);
    workspaceLx.alloc(Lx);
    workspaceLy.alloc(Ly);
    workspaceLx2.alloc(Lx * Lx);
    workspaceLxLy.alloc(Lx * Ly);
    v_0.alloc(Ly);
    v_1.alloc(Ly);

    std::cout << "Iteration:\n\tState length: " << Lx << "; Observation length: " << Ly << "; Sigma points: " << Ls << "\n";
    timer.Record(type);

    // Unscented Transformation of the state
    UnscentedTransformation(xs, x, Pxx, Lx, Ls, workspaceLx2, type);
    timer.Record(type);

    // Evolve and Measure each state given by each sigma point
    pmodel->Evolve(pstate, pcontrol, ExecutionType::Instance, type);
    timer.Record(type);

    pmodel->Evaluate(pmeasure, pstate, ExecutionType::Instance, type);
    timer.Record(type);

    // Calculate new mean and covariance for the state and measure
    Math::Mean(x, xs, wm, Lx, Ls, type);
    pmodel->Evaluate(pmeasure, pstate, ExecutionType::State, type);
    Math::Sub(v_0, ym, y, Ly, type);
    Math::Mean(y, ys, wm, Ly, Ls, type);
    timer.Record(type);

    Math::Iterate(Math::Sub, xs, x, Lx, Ls, Lx, 0, 0, 0, type);
    Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);

    Math::MatMulNWT(0.0, Pxx, 1.0, xs, xs, wc, Lx, Ls, Lx, type);
    Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);
    Math::MatMulNWT(0.0, PxyT, 1.0, ys, xs, wc, Ly, Ls, Lx, type);
    Math::Add(Pxx, Q, Lx * Lx, type);
    Math::Add(Pyy, R, Ly * Ly, type);
    timer.Record(type);

    // Kalman gain calculation by solving: Pyy * (K^T) = Pxy^T
    Math::CholeskySolver(KT, Pyy, PxyT, Ly, Ly, Lx, type);
    timer.Record(type);

    // State Update
    Math::Sub(workspaceLy, ym, y, Ly, type);
    Math::MatMulTN(1.0, x, 1.0, KT, workspaceLy, Lx, Ly, 1, type);
    timer.Record(type);

    Math::MatMulTN(0.0, workspaceLxLy, 1.0, KT, Pyy, Lx, Ly, Ly, type);
    Math::MatMulNN(1.0, Pxx, -1.0, workspaceLxLy, KT, Lx, Ly, Lx, type);
    timer.Record(type);


    // Calculate new observation and covariance
    UnscentedTransformation(xs, x, Pxx, Lx, Ls, workspaceLx2, type);
    pmodel->Evaluate(pmeasure, pstate, ExecutionType::Instance, type);
    Math::Mean(y, ys, wm, Ly, Ls, type);
    Math::Iterate(Math::Sub, ys, y, Ly, Ls, Ly, 0, 0, 0, type);
    Math::MatMulNWT(0.0, Pyy, 1.0, ys, ys, wc, Ly, Ls, Ly, type);

#if ADAPTIVE == 1
    AdaptNoise(x, Pxx, Q, xs, y, Pyy, R, ys, ym, v_0, v_1, PxyT, KT, Lx, Ls, Ly, workspaceLx, workspaceLy, workspaceLx2, workspaceLxLy);
#endif

    v_1.free();
    v_0.free();
    workspaceLxLy.free();
    workspaceLx2.free();
    workspaceLy.free();
    workspaceLx.free();
    KT.free();
    PxyT.free();

    if (type == Type::GPU)
    {
        x.copyDev2Host(Lx);
        Pxx.copyDev2Host(Lx * Lx);
        y.copyDev2Host(Ly);
        Pyy.copyDev2Host(Ly * Ly);
        v_0.copyDev2Host(Ly);
        v_1.copyDev2Host(Ly);
        cudaDeviceSynchronize();
    }
}
