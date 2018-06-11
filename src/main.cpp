#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace Eigen;
using namespace std;

class Func {
public:
    virtual VectorXd f(VectorXd x) = 0;
    virtual MatrixXd jacobian(VectorXd x) = 0;
};

class Func00 : public Func {
    public:
        VectorXd f(VectorXd x);
        MatrixXd jacobian(VectorXd x);
};

VectorXd Func00::f(VectorXd x)
{
    VectorXd v(2);
    // x ^ 2 +     y ^ 2
    // x ^ 2 + 2 * y ^ 2
    v << x[0] * x[0] + 1 * x[1] * x[1],
         x[0] * x[0] + 2 * x[1] * x[1];
    return v;
}

MatrixXd Func00::jacobian(VectorXd x)
{
    MatrixXd m(2,2);
    m << 2 * x[0], 2 * x[1],
         2 * x[0], 4 * x[1];
    return m;
}

class Func01 : public Func {
    public:
        VectorXd f(VectorXd x);
        MatrixXd jacobian(VectorXd x);
};

VectorXd Func01::f(VectorXd x)
{
    VectorXd v(2);
    // (x - 3) ^ 2 +     y ^ 2
    // (x - 3) ^ 2 + 2 * y ^ 2
    auto x3 = (x[0] - 3);
    auto y2 = x[1] * x[1];
    v << x3 * x3  +  1 * y2,
         x3 * x3  +  2 * y2;
    return v;
}

MatrixXd Func01::jacobian(VectorXd x)
{
    MatrixXd m(2,2);
    m << 2 * (x[0] - 3), 2 * x[1],
         2 * (x[0] - 3), 4 * x[1];
    return m;
}

class Func02 : public Func {
    public:
        VectorXd f(VectorXd x);
        MatrixXd jacobian(VectorXd x);
};

VectorXd Func02::f(VectorXd x)
{
    VectorXd v(2);
    // x + 1
    // y + 2
    v << x[0] * x[0] + 1,
         x[1] + 2;
    return v;
}

MatrixXd Func02::jacobian(VectorXd x)
{
    MatrixXd m(2,2);
    m << 2 * x[0], 0,
         0, 1;
    return m;
}

VectorXd step(Func* f, VectorXd x)
{
    // J * (x - x0) = -F(x0)
    auto J = f->jacobian(x);
    auto F = f->f(x);
    VectorXd s = J.colPivHouseholderQr().solve(-F);
    return x + s;
}

int main()
{   
    auto f = new Func01;
    VectorXd x0(2);
    x0 << 211, 10;
    auto s = step(f, x0);

    for (int i = 0; i < 20; i++) {
        cout << s << endl;
        s = step(f, s);
    }
}