#pragma once

#ifndef PAPC_H
#define PAPC_H

#define DataStore_Period 10000

#include "Prox.h"
#include "Derivative.h"
#include "Matrix.h"

/*
PAPC class contains PAPC algorithm
PAPC solves saddle point problem
Problem is :
minimize_{x} maximize_{y}  { f(x) + y^T A x - g(y) } where f differentiable, g CCP
*/

class PAPC {
public:
    Func f = Func();
    Func g = Func();
    Matrix A = Matrix();
    unsigned int p, q;
    double alpha = 0.25;
    double beta = 1.0;
    double gamma = 1.0;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    int SUB_MAX_ITER = 100;

    vector<pair<vector<double>, vector<double>>> data;

    PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q);
    PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _beta, double _gamma);
    PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> x0, vector<double> y0);
    vector<pair<vector<double>, vector<double>>> GetData();
};


inline PAPC::PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q) {
    f = _f;
    g = _g;
    A = _A;
    p = _p;
    q = _q;
}

inline PAPC::PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _beta, double _gamma) {
    f = _f;
    g = _g;
    A = _A;
    p = _p;
    q = _q;
    beta = _beta;
    gamma = _gamma;
}

inline PAPC::PAPC(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
    f = _f;
    g = _g;
    A = _A;
    p = _p;
    q = _q;
    alpha = _alpha;
    beta = _beta;
    gamma = _gamma;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
    SUB_MAX_ITER = _SUB_MAX_ITER;
}

inline void PAPC::solve(vector<double> x0, vector<double> y0) {
    vector<double> x = x0, y = y0;
    vector<double> prev_x = x0, prev_y = y0;
    data.push_back(pair<vector<double>, vector<double>>(x0, y0));

    Matrix bAt = A.Transpose() * beta;
    Matrix rA = A * gamma;

    Derivative derivative(f, p);
    Prox prox_g(g, q, alpha, gamma, threshold, SUB_MAX_ITER);

    double rel_diff;
    const clock_t begin = clock();

    vector<double> tmpx(p), tmpy(q), der(p);

    for (int i = 1 ; i < MAX_ITER ; i++) {
        der = derivative.derivative(x, alpha/100);

        tmpx = bAt * y;
        for (int j = 0 ; j < p ; j++) {
            tmpx.at(j) += beta * der.at(j) - x.at(j);
        }
        tmpy = rA * tmpx;
        for (int j = 0 ; j < q ; j++) {
            tmpy.at(j) = y.at(j) - tmpy.at(j);
        }
        y = prox_g.Proximal(tmpy);

        tmpx = bAt * y;
        for (int j = 0 ; j < p ; j++) {
            x.at(j) -= (tmpx.at(j) + beta * der.at(j));
        }

        rel_diff = 0;
        for (int j = 0 ; j < p ; j++) {
            rel_diff += (prev_x.at(j) - x.at(j)) * (prev_x.at(j) - x.at(j));
        }
        for (int j = 0 ; j < q ; j++) {
            rel_diff += (prev_y.at(j) - y.at(j)) * (prev_y.at(j) - y.at(j));
        }

        prev_x = x;
        prev_y = y;

        if (rel_diff < threshold * threshold) {
            data.push_back(pair<vector<double>, vector<double>>(x, y));
            cout << "PDHG Converged with iterate " << i << endl;
            cout << "PDHG Converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
            cout << "x : ";
            for (int j = 0 ; j < p ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            cout << "y : ";
            for (int j = 0 ; j < q ; j++) {
                cout << y.at(j) << " ";
            }
            cout << endl;
            return;
        }

        if (i % DataStore_Period == 0) {
            data.push_back(pair<vector<double>, vector<double>>(x, y));
            cout << "Iterate : " << i << endl;
        }
    }

    cout << "PAPAC not Converged" << endl;
    cout << "PAPC time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline vector<pair<vector<double>, vector<double> > > PAPC::GetData() {
    return data;
}


#endif //PAPC_H
