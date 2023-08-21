#include "./src/hfe/include/hfe.hpp"
#include "./src/raukf/include/raukf.hpp"
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

    raukf.SetParameters(1e-3, 2.0, 0.0);
    raukf.SetModel(&hfe);
    raukf.SetType(Type::CPU);

    raukf.SetWeight();

    for (int i = 0; i < Lt; i++)
    {
        raukf.Iterate(timer);
    }

    raukf.UnsetModel();
    return 0;
}