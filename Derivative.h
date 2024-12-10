#pragma once

#ifndef DERIVATIVE_H
#define DERIVATIVE_H

#include <cmath>
#include <vector>
#include <functional>

using namespace std;

class Derivative {
public:
    std::function<double(vector<double>)> f;
    unsigned int dim;
    Derivative(std::function<double(vector<double>)> _f, unsigned int _dim) {
        f = _f;
        dim = _dim;
    }
    vector<double> derivative(vector<double> x0, double h);
};

inline vector<double> Derivative::derivative(vector<double> x0, double h) {
    vector<double> x1, x2, ret;
    for (int i = 0 ; i < dim ; i++) {
        x1 = x0;
        x2 = x0;
        x1.at(i) += h;
        x2.at(i) -= h;
        ret.push_back((f(x1) - f(x2)) / (2 * h));
    }
    return ret;
}

#endif //DERIVATIVE_H
