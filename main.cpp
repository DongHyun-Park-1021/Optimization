#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <SDL.h>
#include <float.h>

#include "ADMM.h"

using namespace std;

// f(x, y, z, w) = x^4 + y^4 + z^2 + w^2 - 0.5 * x^2 y^2 - 0.25 xz - 0.25 xw + 0.1 y^3 z - 0.3 y^3 w
double f(vector<double> x) {
    double ret = pow(x[0], 4) + pow(x[1], 4) + pow(x[2], 2) + pow(x[3], 2) - 0.5 * pow(x[0] * x[1], 2);
    ret -= (0.25 * x[0] * x[2] + 0.25 * x[0] * x[3] - 0.1 * pow(x[1], 3) * x[2] + 0.3 * pow(x[1], 3) * x[3]);
    return ret;
}

// g(x, y, z) = x^4 + y^4 + z^4 + sin(x) + 2sin(xy) + 4sin(xyz)
double g(vector<double> x) {
    double ret = pow(x[0], 4) + pow(x[1], 4) + pow(x[2], 4);
    ret += (sin(x[0]) + 2 * sin(x[0] * x[1]) + 4 * sin(x[0] * x[1] * x[2]));
    return ret;
}


int main() {
    int p = 4, q = 3, r = 2;
    Func F = Func(f, p);
    Func G = Func(g, q);

    vector<vector<double>> a, b;
    a.push_back({1, 2, -1, -1});
    a.push_back({0, -1, 1, -3});

    b.push_back({-1, 0.5, 0});
    b.push_back({0, 1, -2});

    Matrix A = Matrix(r, p, a);
    Matrix B = Matrix(r, q, b);
    vector<double> c = {1, -1};
    ADMM admm = ADMM(F, G, A, B, c, p, q, r, 0.0001, 0.001, 0.001, 0.00001, 1000000, 5000);

    vector<double> x0 = {0, 0, 0, 0};
    vector<double> y0 = {0, 0, 0};

    vector<pair<vector<double>, vector<double>>> ret_admm;
    admm.solve(x0, y0);

    vector<pair<vector<double>, vector<double>>> Data = admm.GetData();

    vector<double> Ax(r), By(r), u(r);
    /*
    for (int i = 0 ; i < Data.size(); i++) {
        cout << "x : ";
        for (int j = 0 ; j < p ; j++) {
            cout << Data.at(i).first.at(j) << " ";
        }
        cout << endl;
        cout << "y : ";
        for (int j = 0 ; j < q ; j++) {
            cout << Data.at(i).second.at(j) << " ";
        }
        cout << endl;
        cout << "f(x) + g(y) = " << f(Data.at(i).first) + g(Data.at(i).second) << endl;

        Ax = A * Data.at(i).first;
        By = B * Data.at(i).second;

        cout << "Ax + By - c = ";
        for (int j = 0 ; j < r ; j++) {
            u.at(j) = Ax.at(j) + By.at(j) - c.at(j);
            cout << u.at(j) << " ";
        }
        cout << endl;
    }
    */
}
