#pragma once

#ifndef PGM_H
#define PGM_H

#define DataStore_Period 10000

#include "Prox.h"
#include "Derivative.h"

/*
PGM class contains Proximal Gradient Method Algorithm
Problem is :
minimize f(x) + g(x) where f is differentiable and f, g CCP
*/

class PGM {
public:
    Func f = Func();
    Func g = Func();
    unsigned int dim;
    double alpha = 0.25;
    double beta = 1.0;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    int SUB_MAX_ITER = 100;

    vector<vector<double>> data;

    PGM(Func _f, Func _g, unsigned int _dim);
    PGM(Func _f, Func _g, unsigned int _dim, double _beta);
    PGM(Func _f, Func _g, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER);

    void solve(vector<double> x0);
    vector<vector<double>> GetData();
};

inline PGM::PGM(Func _f, Func _g, unsigned int _dim) {
    f = _f;
    g = _g;
    dim = _dim;
}

inline PGM::PGM(Func _f, Func _g, unsigned int _dim, double _beta) {
    f = _f;
    g = _g;
    dim = _dim;
    beta = _beta;
}


inline PGM::PGM(Func _f, Func _g, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER, int _SUB_MAX_ITER) {
    f = _f;
    g = _g;
    dim = _dim;
    alpha = _alpha;
    beta = _beta;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
    SUB_MAX_ITER = _SUB_MAX_ITER;
}

inline void PGM::solve(vector<double> x0) {
    vector<double> x(dim);
    x = x0;
    vector<double> prev(dim), der(dim);
    data.push_back(x);

    Derivative derivative(f, dim);
    Prox prox(g, dim, alpha, beta, threshold, SUB_MAX_ITER);

    double rel_diff;
    vector<double> y = x;
    const clock_t begin = clock();

    for (int i = 1 ; i < MAX_ITER ; i++) {
        rel_diff = 0;

        der = derivative.derivative(x, alpha/100);
        for (int j = 0 ; j < dim ; j++) {
            prev.at(j) = x.at(j);
            y.at(j) = x.at(j) - der.at(j) * alpha;
        }

        x = prox.Proximal(y);

        for (int j = 0 ; j < dim ; j++) {
            rel_diff += (prev.at(j) - x.at(j)) * (prev.at(j) - x.at(j));
        }
        rel_diff /= dim;

        if (i % DataStore_Period == 0) {
            data.push_back(x);
            cout << "Iterate : " << i << endl;
        }

        if (rel_diff < threshold * threshold) {
            cout << "PGM Converged with iterate " << i << endl;
            cout << "PGM Converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;

            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }
    }

    cout << "PGM not Converged" << endl;
    cout << "PGM time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline vector<vector<double> > PGM::GetData() {
    return data;
}


#endif //PGM_H
