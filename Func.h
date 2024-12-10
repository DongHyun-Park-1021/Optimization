#pragma once

#ifndef FUNCTION_H
#define FUNCTION_H
#include <functional>
#include <vector>

using namespace std;

/*
Func class determines function.
In the algorithm, two kind of function is needed :

(1) classical function
(2) function determining region : in this case \delta_{f <= 0} is the desired region
*/

class Func {
public:
    std::function<double(vector<double>)> f;
    unsigned int dim;
    bool isRegion = false;
    vector<double> feasible;

    Func();
    Func(std::function<double(vector<double>)> _f, unsigned int _dim);
    Func(std::function<double(vector<double>)> _f, unsigned int _dim, bool _isRegion, vector<double> _feasible);
    double F(vector<double> x);
};

inline Func::Func() {

}

inline Func::Func(std::function<double(vector<double>)> _f, unsigned int _dim) {
    f = _f;
    dim = _dim;
}


inline Func::Func(std::function<double(vector<double>)> _f, unsigned int _dim, bool _isRegion, vector<double> _feasible) {
    f = _f;
    dim = _dim;
    isRegion = _isRegion;
    feasible = _feasible;
}

inline double Func::F(vector<double> x) {
    return f(x);
}

#endif //FUNCTION_H
