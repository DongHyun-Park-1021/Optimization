#pragma once

#ifndef PROX_H
#define PROX_H

#include <cmath>
#include <functional>
#include "Derivative.h"

using namespace std;

/*
Prox_diff class contains Problem set : Prox_f (y) = minimize ( f(x) + 1/2||x-y||^2 )
*/

class Prox {
public:
    std::function<double(vector<double>)> f;
    unsigned int dim;
    double alpha = 0.25, threshold = 0.00001;
    int MAX_ITER = 100;
    double beta = 1.0;

    Prox(std::function<double(vector<double>)> _f, unsigned int _dim);
    Prox(std::function<double(vector<double>)> _f, unsigned int _dim, double _beta);
    Prox(std::function<double(vector<double>)> _f, int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER);

    vector<double> Proximal(vector<double> x);
};

inline Prox::Prox(std::function<double(vector<double>)> _f, unsigned int _dim) {
    f = _f;
    dim = _dim;
}
inline Prox::Prox(std::function<double(vector<double>)> _f, unsigned int _dim, double _beta) {
    f = _f;
    dim = _dim;
    beta = _beta;
}
inline Prox::Prox(std::function<double(vector<double>)> _f, int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER){
    f = _f;
    beta = _beta;
    dim = _dim;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline vector<double> Prox::Proximal(vector<double> x) {
    vector<double> y(dim);
    y = x;
    vector<double> prev(dim);
    Derivative derivative(f, dim);
    double rel_diff;

    vector<double> der;
    for (int i = 1 ; i < MAX_ITER ; i++) {
        rel_diff = 0;
        der = derivative.derivative(y, alpha);
        for (int j = 0 ; j < dim ; j++) {
            prev.at(j) = x.at(j);
            y.at(j) -= (der.at(j) + (y.at(j) - x.at(j))/beta) * alpha;
            rel_diff += (prev.at(j) - x.at(j)) * (prev.at(j) - x.at(j));
        }

        if (rel_diff < threshold * threshold) {
            return y;
        }
    }

    cout << "Proximal not Converged" << endl;
    return y;
}

#endif //PROX_H
