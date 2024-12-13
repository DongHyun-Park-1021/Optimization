#pragma once

#ifndef PDHG_H
#define PDHG_H

#define DataStore_Period 10000

#include "Prox.h"
#include "Matrix.h"

/*
PDHG class contains PDHG algorithm
PDHG solves saddle point problem
Problem is :
minimize_{x} maximize_{y}  { f(x) + y^T A x - g(y) } where f, g CCP
*/

class PDHG {
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

    PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q);
    PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _beta, double _gamma);
    PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> x0, vector<double> y0);
    vector<pair<vector<double>, vector<double>>> GetData();
};

inline PDHG::PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q) {
    f = _f;
    g = _g;
    A = _A;
    p = _p;
    q = _q;
}

inline PDHG::PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _beta, double _gamma) {
    f = _f;
    g = _g;
    A = _A;
    p = _p;
    q = _q;
    beta = _beta;
    gamma = _gamma;
}

inline PDHG::PDHG(Func _f, Func _g, Matrix _A, unsigned int _p, unsigned int _q, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
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

inline void PDHG::solve(vector<double> x0, vector<double> y0) {
    vector<double> x = x0, y = y0;
    vector<double> prev_x = x0, prev_y = y0;
    data.push_back(pair<vector<double>, vector<double>>(x0, y0));

    Matrix bAt = A.Transpose() * beta;
    Matrix rA = A * gamma;

    Prox prox_f(f, p, alpha, beta, threshold, SUB_MAX_ITER);
    Prox prox_g(g, q, alpha, gamma, threshold, SUB_MAX_ITER);

    double rel_diff;
    const clock_t begin = clock();

    vector<double> tmpx(p), tmpy(q);

    for (int i = 1 ; i < MAX_ITER ; i++) {
        tmpx = bAt * y;
        for (int j = 0 ; j < p ; j++) {
            x.at(j) -= tmpx.at(j);
        }

        x = prox_f.Proximal(x);

        for (int j = 0 ; j < p ; j++) {
            tmpx.at(j) = 2 * x.at(j) - prev_x.at(j);
        }
        tmpy = rA * tmpx;
        for (int j = 0 ; j < q ; j++) {
            y.at(j) += tmpy.at(j);
        }

        y = prox_g.Proximal(y);

        rel_diff = 0;
        for (int j = 0 ; j < p ; j++) {
            rel_diff += (prev_x.at(j) - x.at(j)) * (prev_x.at(j) - x.at(j));
        }
        for (int j = 0 ; j < q ; j++) {
            rel_diff += (prev_y.at(j) - y.at(j)) * (prev_y.at(j) - y.at(j));
        }
        rel_diff /= (p + q);

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

    cout << "PDHG not Converged" << endl;
    cout << "PDHG time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline vector<pair<vector<double>, vector<double>>> PDHG::GetData() {
    return data;
}

#endif //PDHG_H
