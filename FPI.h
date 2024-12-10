#pragma once

#ifndef FPI_H
#define FPI_H

#include <cmath>
#include <functional>
#include <vector>
#include "Func.h"
#include "Derivative.h"

using namespace std;

/*
FPI class contains Problem set : minimize f(x) where x in R^d
*/

class FPI {
public:
    Func f = Func();
    unsigned int dim;
    vector<vector<double>> data;
    double alpha = 0.25;
    int MAX_ITER = 100;
    double threshold = 0.00001;

    FPI(Func _f, int _dim);
    FPI(Func _f, int _dim, double _alpha, int _MAX_ITER, double _threshold);

    void solve(vector<double> x0, double alpha);
    vector<vector<double>> Get_data();
private:
    vector<double> derivative(vector<double> x0, double alpha);
};

inline FPI::FPI(Func _f, int _dim) {
    f = _f;
    dim = _dim;
}
inline FPI::FPI(Func _f, int _dim, double _alpha, int _MAX_ITER, double _threshold) {
    f = _f;
    dim = _dim;
    alpha = _alpha;
    MAX_ITER = _MAX_ITER;
    threshold = _threshold;
}


inline void FPI::solve(vector<double> x0, double alpha) {
    vector<double> x = x0;
    vector<double> prev;
    Derivative derivative(f, dim);

    double rel_diff;
    data.push_back(x);

    vector<double> der;
    for (int i = 1 ; i < MAX_ITER ; i++) {
        rel_diff = 0;
        der = derivative.derivative(x, alpha/100);
        for (int j = 0 ; j < dim ; j++) {
            prev.at(j) = x.at(j);
            x.at(j) -= der.at(j) * alpha;
            rel_diff += (prev.at(j) - x.at(j)) * (prev.at(j) - x.at(j));
        }
        data.push_back(x);

        if (rel_diff < threshold * threshold) {
            cout << "FPI Converged with iterate " << i << endl;
            for (int j = 0 ; j < dim ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }
    }

    cout << "FPI not Converged" << endl;
}
inline vector<vector<double>> FPI::Get_data() {
    return data;
}

#endif //FPI_H
