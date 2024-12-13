#pragma once

#ifndef ADMM_H
#define ADMM_H

#define DataStore_Period 10000

#include "Prox.h"
#include "Derivative.h"
#include "Matrix.h"

/*
ADMM class contains Linearized ADMM method
Problem is :
minimize f(x) + g(x) where f, g CCP and Ax + By = c
*/

class ADMM {
public:
    Func f = Func();
    Func g = Func();
    Matrix A = Matrix();
    Matrix B = Matrix();
    vector<double> c;
    unsigned int p, q, r;
    double alpha = 0.25;
    double beta = 1.0;
    double gamma = 1.0;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    int SUB_MAX_ITER = 100;

    vector<pair<vector<double>, vector<double>>> data;

    ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r);
    ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r, double _beta, double _gamma);
    ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> x0, vector<double> y0);
    vector<pair<vector<double>, vector<double>>> GetData();
};

inline ADMM::ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r) {
    f = _f;
    g = _g;
    A = _A;
    B = _B;
    c = _c;
    p = _p;
    q = _q;
    r = _r;
}

inline ADMM::ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r, double _beta, double _gamma) {
    f = _f;
    g = _g;
    A = _A;
    B = _B;
    c = _c;
    p = _p;
    q = _q;
    r = _r;
    beta = _beta;
    gamma = _gamma;
}

inline ADMM::ADMM(Func _f, Func _g, Matrix _A, Matrix _B, vector<double> _c, unsigned int _p, unsigned int _q, unsigned int _r, double _alpha, double _beta, double _gamma, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
    f = _f;
    g = _g;
    A = _A;
    B = _B;
    c = _c;
    p = _p;
    q = _q;
    r = _r;
    beta = _beta;
    gamma = _gamma;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
    SUB_MAX_ITER = _SUB_MAX_ITER;
}

inline void ADMM::solve(vector<double> x0, vector<double> y0) {
    vector<double> x = x0, y = y0, u(r);
    for (int i = 0 ; i < r ; i++) { u[i] = 0; }
    vector<double> prev_x = x0, prev_y = y0;
    data.push_back(pair<vector<double>, vector<double>>(x0, y0));

    Matrix aA = A * alpha;
    Matrix aB = B * alpha;
    Matrix bAt = A.Transpose() * beta;
    Matrix rBt = B.Transpose() * gamma;

    Prox prox_f(f, p, alpha, beta, threshold, SUB_MAX_ITER);
    Prox prox_g(g, q, alpha, gamma, threshold, SUB_MAX_ITER);

    double rel_diff;
    const clock_t begin = clock();

    vector<double> aAx(r), aBy(r), tmp(r), tmpx(p), tmpy(q);

    for (int i = 1 ; i < MAX_ITER ; i++) {
        aAx = aA * x;
        aBy = aB * y;
        for (int j = 0 ; j < r ; j++) {
            tmp.at(j) = u.at(j) + aAx.at(j) + aBy.at(j) - alpha * c.at(j);
        }
        tmpx = bAt * tmp;
        for (int j = 0 ; j < p ; j++) {
            x.at(j) -= tmpx.at(j);
        }

        x = prox_f.Proximal(x);

        aAx = aA * x;
        for (int j = 0 ; j < r ; j++) {
            tmp.at(j) = u.at(j) + aAx.at(j) + aBy.at(j) - alpha * c.at(j);
        }
        tmpy = rBt * tmp;
        for (int j = 0 ; j < q ; j++) {
            y.at(j) -= tmpy.at(j);
        }

        y = prox_g.Proximal(y);

        aBy = aB * y;
        for (int j = 0 ; j < r ; j++) {
            u.at(j) += (aAx.at(j) + aBy.at(j) - alpha * c.at(j));
        }

        rel_diff = 0;
        for (int j = 0 ; j < p ; j++) {
            rel_diff += (prev_x.at(j) - x.at(j)) * (prev_x.at(j) - x.at(j));
        }
        for (int j = 0 ; j < q ; j++) {
            rel_diff += (prev_y.at(j) - y.at(j)) * (prev_y.at(j) - y.at(j));
        }
        for (int j = 0 ; j < r ; j++) {
            rel_diff += (aAx.at(j) + aBy.at(j) - alpha * c.at(j)) * (aAx.at(j) + aBy.at(j) - alpha * c.at(j)) / (alpha * alpha);
        }
        rel_diff /= (p + q + r);

        prev_x = x;
        prev_y = y;

        if (rel_diff < threshold * threshold) {
            data.push_back(pair<vector<double>, vector<double>>(x, y));
            cout << "ADMM Converged with iterate " << i << endl;
            cout << "ADMM Converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
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
            cout << "f(x) + g(y) = " << f.F(x) + g.F(y) << endl;

            cout << "Ax + By - c = ";
            for (int j = 0 ; j < r ; j++) {
                cout << aAx.at(j)/alpha + aBy.at(j)/alpha - c.at(j) << " ";
            }
            cout << endl;
            return;
        }

        if (i % DataStore_Period == 0) {
            data.push_back(pair<vector<double>, vector<double>>(x, y));
            cout << "Iterate : " << i << endl;
        }
    }

    cout << "ADMM not Converged" << endl;
    cout << "ADMM time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline vector<pair<vector<double>, vector<double>>> ADMM::GetData() {
    return data;
}


#endif //ADMM_H
