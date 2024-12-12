#pragma once

#ifndef PROX_H
#define PROX_H

#include <cmath>
#include <functional>
#include "Derivative.h"
#include "Func.h"

using namespace std;

/*
Prox_diff class contains Problem set : Prox_f (y) = minimize ( f(x) + 1/(2\beta) ||x-y||^2 )
*/

class Prox {
public:
    Func f = Func();
    unsigned int dim;
    double alpha = 0.25, threshold = 0.001;
    int MAX_ITER = 100;
    double beta = 1.0;

    Prox(Func _f, unsigned int _dim);
    Prox(Func _f, unsigned int _dim, double _beta);
    Prox(Func _f, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER);
    Prox(Func _f, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER);

    vector<double> Proximal(vector<double> x);
private:
    vector<double> Proximal_diff(vector<double> x);
    vector<double> Proximal_region(vector<double> x);
};

inline Prox::Prox(Func _f, unsigned int _dim) {
    f = _f;
    dim = _dim;
}

inline Prox::Prox(Func _f, unsigned int _dim, double _beta) {
    f = _f;
    dim = _dim;
    beta = _beta;
}

inline Prox::Prox(Func _f, unsigned int _dim, double _alpha, double _threshold, int _MAX_ITER){
    f = _f;
    dim = _dim;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline Prox::Prox(Func _f, unsigned int _dim, double _alpha, double _beta, double _threshold, int _MAX_ITER){
    f = _f;
    beta = _beta;
    dim = _dim;
    alpha = _alpha;
    threshold = _threshold;
    MAX_ITER = _MAX_ITER;
}

inline vector<double> Prox::Proximal(vector<double> x) {
    if (f.isRegion) {
        return Proximal_region(x);
    } else {
        return Proximal_diff(x);
    }
}

inline vector<double> Prox::Proximal_diff(vector<double> x) {
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
            prev.at(j) = y.at(j);
            y.at(j) -= (der.at(j) + (y.at(j) - x.at(j))/beta) * alpha;
            rel_diff += (prev.at(j) - y.at(j)) * (prev.at(j) - y.at(j));
        }

        if (rel_diff < threshold * threshold) {
            return y;
        }
    }

    cout << "Proximal not Converged" << endl;
    /*
    for (int j = 0 ; j < dim ; j++) {
        cout << y.at(j) << " ";
    }
    cout << endl;
    cout << "prev : ";
    for (int j = 0 ; j < dim ; j++) {
        cout << prev.at(j) << " ";
    }
    cout << endl;
    */
    return y;
}

inline vector<double> Prox::Proximal_region(vector<double> x) {
    if (f.F(x) <= 0) return x;

    vector<double> y = f.feasible, y1 = f.feasible, z = x, w = x, y1_prev = f.feasible;
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

            if (f.F(w) <= 0) {
                y = w;
            }else {
                z = w;
            }

            dist /= 4;
        }

        der = derivative.derivative(y, alpha/100);

        length = 0;
        inner_product = 0;
        for (int j = 0; j < dim; j++) {
            length += der.at(j) * der.at(j);
            inner_product += der.at(j) * (x.at(j) - y.at(j));
        }

        length = sqrt(length);
        inner_product = inner_product / length;

        for (int j = 0 ; j < dim ; j++) {
            y.at(j) = y.at(j) + alpha * (x.at(j) - y.at(j) - der.at(j) / length * inner_product); // y_new = y + (x - y) - der / length * inner_product
        }

        y1 = y;

        dist = 0;
        for (int j = 0 ; j < dim ; j++) {
            dist += ((y1.at(j) - y1_prev.at(j)) * (y1.at(j) - y1_prev.at(j)));
        }

        if (dist < threshold * threshold) {
            return y1;
        }

        y1_prev = y;

        if (i < MAX_ITER - 1) {
            if (f.F(y) <= 0) {
                z = x;
            }else {
                z = y;
                y = f.feasible;
            }
        }
    }

    cout << "Proximal not Converged" << endl;
    /*
    for (int j = 0 ; j < dim ; j++) {
        cout << y1.at(j) << " ";
    }
    cout << endl;
    cout << "prev : ";
    for (int j = 0 ; j < dim ; j++) {
        cout << y1_prev.at(j) << " ";
    }
    cout << endl;
    */
    return y1;
}


#endif //PROX_H
