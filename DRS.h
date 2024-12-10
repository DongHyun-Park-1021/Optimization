#pragma once

#ifndef DRS_H
#define DRS_H

#include "Prox.h"

/*
DRS class contains DRS Algorithm
Problem is :
minimize f(x) + g(x) where f, g are CCP (no need to be differentiable)
*/

class DRS {
public:
    std::function<double(vector<double>)> f;
    std::function<double(vector<double>)> g;
    unsigned int dim;
    double alpha = 0.25;
    double threshold = 0.00001;
    int MAX_ITER = 100;

    vector<vector<double>> data;

    DRS(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim);
    DRS(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER);

    void solve(vector<double> z0);
    vector<vector<double>> GetData();
};

inline DRS::DRS(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim) {
    f = _f;
    g = _g;
    dim = _dim;
}

inline DRS::DRS(std::function<double(vector<double>)> _f, std::function<double(vector<double>)> _g, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER) {
    f = _f;
    g = _g;
    dim = _dim;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline void DRS::solve(vector<double> z0) {
    vector<double> z(dim);
    z = z0;
    vector<double> x(dim), x_half(dim), y(dim);
    vector<double> prev(dim);

    Prox prox_f(f, dim, alpha, alpha, threshold, MAX_ITER);
    Prox prox_g(g, dim, alpha, alpha, threshold, MAX_ITER);

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

inline vector<vector<double>> DRS::GetData() {
    return data;
}

#endif //DRS_H
