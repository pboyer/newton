#include <iostream>
#include <Eigen/Dense>
#include <math.h>

using namespace Eigen;
using namespace std;

VectorXd f(VectorXd x)
{
    VectorXd v(2);
    // x ^ 2 +     y ^ 2
    // x ^ 2 + 2 * y ^ 2
    v << x[0] * x[0] + 1 * x[1] * x[1],
         x[0] * x[0] + 2 * x[1] * x[1];
    return v;
}

MatrixXd jacobian(VectorXd x)
{
    MatrixXd m(2,2);
    m << 2 * x[0], 2 * x[1],
         2 * x[0], 4 * x[1];
    return m;
}

VectorXd step(VectorXd x)
{
    // J * (x - x0) = -F(x0)
    auto J = jacobian(x);
    auto F = f(x);
    VectorXd s = J.colPivHouseholderQr().solve(-F);
    return x + s;
}

int main()
{   
    VectorXd x0(2);
    x0 << 5, 2;
    auto s = step(x0);

    for (int i = 0; i < 10; i++) {
        cout << s << endl;
        s = step(s);
    }
}