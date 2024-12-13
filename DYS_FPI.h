#pragma once

#ifndef DYS_FPI_H
#define DYS_FPI_H

#include "Func.h"
#include "Derivative.h"
#include "Prox.h"


/*
DRS_FPI class contains DYS (Davis - Yin Splitting) Algorithm
Problem is :
minimize f(x) + g(x) + h(x) where f, g are CCP, h is differentiable
*/


class DYS_FPI {
public:
    Func f = Func();
    Func g = Func();
    Func h = Func();
    unsigned int dim;
    double alpha = 0.25;
    double beta = 1.0;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    int SUB_MAX_ITER = 100;

    vector<vector<double>> data;

    DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim);
    DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim, double _beta);
    DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> x0);
    vector<vector<double>> GetData();
};

inline DYS_FPI::DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim) {
    f = _f;
    g = _g;
    h = _h;
    dim = _dim;
}

inline DYS_FPI::DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim, double _beta) {
    f = _f;
    g = _g;
    h = _h;
    dim = _dim;
    beta = _beta;
}

inline DYS_FPI::DYS_FPI(Func _f, Func _g, Func _h, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
    f = _f;
    g = _g;
    h = _h;
    dim = _dim;
    beta = _beta;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
    SUB_MAX_ITER = _SUB_MAX_ITER;
}

inline void DYS_FPI::solve(vector<double> z0) {
    vector<double> z(dim);
    z = z0;
    vector<double> x(dim), x_half(dim), y(dim), der(dim);
    vector<double> prev(dim);

    Prox prox_f(f, dim, alpha, beta, threshold, SUB_MAX_ITER);
    Prox prox_g(g, dim, alpha, beta, threshold, SUB_MAX_ITER);
    Derivative der_h(h, dim);

    double rel_diff;
    const clock_t begin = clock();

    for (int i = 0 ; i < MAX_ITER ; i++) {
        rel_diff = 0;
        x_half = prox_g.Proximal(z);
        der = der_h.derivative(x_half, alpha/100);

        for (int j = 0 ; j < dim ; j++) {
            y.at(j) = 2 * x_half.at(j) - z.at(j) - beta * der.at(j);
            if (i > 0) {
                prev.at(j) = x.at(j);
            }
        }

        x = prox_f.Proximal(y);

        for (int j = 0 ; j < dim ; j++) {
            z.at(j) += (x.at(j) - x_half.at(j));
        }
        if (i == 0) {
            cout << "Initial x : ";
            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
        }
        if (i > 0) {
            for (int j = 0 ; j < dim ; j++) {
                rel_diff += (prev.at(j) - x.at(j)) * (prev.at(j) - x.at(j));
            }
        }
        rel_diff /= dim;

        data.push_back(x);

        if (rel_diff < threshold * threshold && i > 0) {
            cout << "DYS Converged with iterate " << i << endl;
            cout << "DYS Converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;

            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }
    }

    cout << "DYS not Converged" << endl;
    cout << "DYS time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline vector<vector<double>> DYS_FPI::GetData() {
    return data;
}



#endif //DYS_FPI_H
