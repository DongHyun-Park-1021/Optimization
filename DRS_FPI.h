#pragma once

#ifndef DRS_FPI_H
#define DRS_FPI_H

#include "Prox.h"
#include "Func.h"

/*
DRS_FPI class contains DRS Algorithm
Problem is :
minimize f(x) + g(x) where f, g are CCP
*/

class DRS_FPI {
public:
    Func f = Func();
    Func g = Func();
    unsigned int dim;
    double alpha = 0.25;
    double beta = 1;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    int SUB_MAX_ITER = 100;

    vector<vector<double>> data;

    DRS_FPI(Func _f, Func _g, unsigned int _dim);
    DRS_FPI(Func _f, Func _g, unsigned int _dim, double _beta);
    DRS_FPI(Func _f, Func _g, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> z0);
    vector<vector<double>> GetData();
};

inline DRS_FPI::DRS_FPI(Func _f, Func _g, unsigned int _dim) {
    f = _f;
    g = _g;
    dim = _dim;
}

inline DRS_FPI::DRS_FPI(Func _f, Func _g, unsigned int _dim, double _beta) {
    f = _f;
    g = _g;
    dim = _dim;
    beta = _beta;
}


inline DRS_FPI::DRS_FPI(Func _f, Func _g, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
    f = _f;
    g = _g;
    dim = _dim;
    alpha = _alpha;
    beta = _beta;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
    SUB_MAX_ITER = _SUB_MAX_ITER;
}

inline void DRS_FPI::solve(vector<double> z0) {
    vector<double> z(dim);
    z = z0;
    vector<double> x(dim), x_half(dim), y(dim);
    vector<double> prev(dim);

    Prox prox_f(f, dim, alpha, beta, threshold, SUB_MAX_ITER);
    Prox prox_g(g, dim, alpha, beta, threshold, SUB_MAX_ITER);

    double rel_diff;

    for (int i = 0 ; i < MAX_ITER ; i++) {
        rel_diff = 0;
        x_half = prox_g.Proximal(z);

        for (int j = 0 ; j < dim ; j++) {
            y.at(j) = 2 * x_half.at(j) - z.at(j);
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

        data.push_back(x);

        if (rel_diff < threshold * threshold && i > 0) {
            cout << "DRS Converged with iterate " << i << endl;
            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }
    }

    cout << "DRS not Converged" << endl;
}

inline vector<vector<double>> DRS_FPI::GetData() {
    return data;
}

#endif //DRS_FPI_H
