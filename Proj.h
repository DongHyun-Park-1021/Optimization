#pragma once

#ifndef PROJ_H
#define PROJ_H

#include <vector>
#include <functional>
#include <vector>
#include <vector>
#include <vector>
#include "Derivative.h"

/*
Proj class contains Projecting Algorithm for convex closed set
Problem is : minimize ||x-y|| for y in C
We especially consider the problem where C is expressed as f(y) <= 0, and assume this set is convex closed
*/

class Proj {
public:
    vector<double> feasible;
    std::function<double(vector<double>)> f;
    unsigned int dim;
    double alpha = 0.25;
    double threshold = 0.00001;
    int MAX_ITER = 100;
    vector<vector<double>> data;

    Proj(vector<double> _feasible, std::function<double(vector<double>)> _f, unsigned int _dim);
    Proj(vector<double> _feasible, std::function<double(vector<double>)> _f, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER);

    vector<double> Project(vector<double> x);
    vector<vector<double>> GetData();
};

inline Proj::Proj(vector<double> _feasible, std::function<double(vector<double>)> _f, unsigned int _dim) {
    feasible = _feasible;
    dim = _dim;
    f = _f;
}

inline Proj::Proj(vector<double> _feasible, std::function<double(vector<double>)> _f, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER) {
    feasible = _feasible;
    dim = _dim;
    f = _f;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline vector<double> Proj::Project(vector<double> x) {
    if (f(x) <= 0) return x;

    vector<double> y = feasible, z = x, w = x, y_prev = feasible;
    double dist = 0;
    Derivative derivative = Derivative(f, dim);
    vector<double> der;
    double length, inner_product;

    for (int i = 0 ; i < MAX_ITER ; i++) {
        dist = 0;
        for (int j = 0; j < dim; j++) {
            dist += (y.at(j) - z.at(j)) * (y.at(j) - z.at(j));
        }

        while (dist > threshold * threshold) {
            for (int j = 0; j < dim; j++) {
                w.at(j) = (z.at(j) + y.at(j)) / 2;
            }

            if (f(w) <= 0) {
                y = w;
            }else {
                z = w;
            }

            dist /= 4;
            i++;
        }

        data.push_back(y);

        der = derivative.derivative(y, alpha/100);

        length = 0;
        inner_product = 0;
        for (int j = 0; j < dim; j++) {
            length += der.at(j) * der.at(j);
            inner_product += der.at(j) * (y.at(j) - x.at(j));
        }
        length = sqrt(length);
        inner_product = inner_product / length;

        dist = 0;
        for (int j = 0 ; j < dim ; j++) {
            y.at(j) = x.at(j) - der.at(j) / length * inner_product; // y_new = y + (x - y) - der / length * inner_product
            dist += (y.at(j) - y_prev.at(j)) * (y.at(j) - y_prev.at(j));
        }

        data.push_back(y);

        if (dist < threshold * threshold) {
            cout << "Projection converged with iterate " << i << endl;
            for (int j = 0 ; j < dim ; j++) {
                cout << y.at(j) << " ";
            }
            cout << endl;
            return y;
        }

        if (f(y) <= 0) {
            y_prev = y;
            z = x;
        }else {
            z = y;
            y_prev = y;
            y = feasible;
        }
    }
    cout << "Projection not converged" << endl;
    return y;
}

inline vector<vector<double>> Proj::GetData() {
    return data;
}




#endif //PROJ_H
