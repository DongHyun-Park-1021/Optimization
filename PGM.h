#pragma once

#ifndef PGM_H
#define PGM_H

#include "Prox.h"
#include "Derivative.h"

/*
PGM class contains Proximal Gradient Method Algorithm
Problem is :
minimize f(x) + g(x) where f is differentiable and f, g CCP
*/

class PGM {
public:
    std::function<double(vector<double>)> f;
    std::function<double(vector<double>)> g;
    unsigned int dim;
    double alpha = 0.25;
    double threshold = 0.00001;
    int MAX_ITER = 100;

    vector<vector<double>> data;

    PGM(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim);
    PGM(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER);

    void solve(vector<double> x0);
    vector<vector<double>> GetData();
};

inline PGM::PGM(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim) {
    f = _f;
    g = _g;
    dim = _dim;
}

inline PGM::PGM(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER) {
    f = _f;
    g = _g;
    dim = _dim;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline void PGM::solve(vector<double> x0) {
    vector<double> x(dim);
    x = x0;
    vector<double> prev(dim), der(dim);
    data.push_back(x);

    Derivative derivative(f, dim);
    Prox prox(g, dim, alpha, alpha, threshold, MAX_ITER);

    double rel_diff;
    vector<double> y = x;

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
        data.push_back(x);

        if (rel_diff < threshold * threshold) {
            cout << "PGM Converged with iterate " << i << endl;
            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }
    }

    cout << "PGM not Converged" << endl;
}

inline vector<vector<double> > PGM::GetData() {
    return data;
}


#endif //PGM_H
