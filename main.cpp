#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <SDL.h>
#include <float.h>

#include "Matrix.h"
#include "Condat_Vu.h"
#include "PD3O.h"

using namespace std;

unsigned int dim1 = 100;
unsigned int dim2 = 100;
int m = 2;

// f(x1, ... xn) = (log( sum_{i=1}^{m} exp(a1i x1 + ... + ani xn) ) ) / (sum_{i=1}^{m} 1/(1 + bi exp(xi)) )
double f(vector<double> x) {
    double dom = 0, temp;
    for (int i = 0 ; i < m; i++) {
        temp = 0;
        for (int j = 0; j < dim1; j++) {
            temp += (((i+1) * (j+1)) * x[j] / ((i + j + 1) * (i + j + 1)));
        }
        temp += (1/(i+1));
        dom += exp(temp);
    }
    dom = log(dom);

    double quotient = 0;
    for (int i = 0 ; i < dim1 ; i++) {
        quotient += 1/(1 + (1/(i+1)) * exp(x[i]));
    }
    return dom / quotient;
}

// g(x1, ... xn) = |x1| + ... + |xn| - 1
double g(vector<double> x) {
    double ret = 0;
    for (int i = 0 ; i < dim2 ; i++) {
        ret += abs(x[i] - 1/sqrt(i+1))/sqrt(i+1);
    }
    return ret - 1;
}

// h(x1, ... xn) = x1^2 + ... + xn^2 - 1
double h(vector<double> x) {
    double ret = 0;
    for (int i = 0 ; i < dim1 ; i++) {
        ret += x[i] * x[i];
    }
    return ret;
}



int main() {
    vector<double> init1, init2;
    for (int i = 0 ; i < dim1 ; i++) {
        init1.push_back(3.0 * rand()/(double)RAND_MAX);
    }
    for (int i = 0 ; i < dim2 ; i++) {
        init2.push_back(3.0 * rand()/(double)RAND_MAX);
    }
    Func F = Func(f, dim1);
    Func G = Func(g, dim2);
    Func H = Func(h, dim1);

    vector<vector<double>> a;
    for (int i = 0 ; i < dim2 ; i++) {
        vector<double> temp;
        for (int j = 0; j < dim1; j++) {
            if (rand()/(double)RAND_MAX < 0.5) {
                temp.push_back(3.0 * rand()/(double)RAND_MAX);
            }else {
                temp.push_back((-3.0) * rand()/(double)RAND_MAX);
            }
        }
        a.push_back(temp);
    }
    Matrix A = Matrix(dim2, dim1, a);

    double alpha = 0.0001;
    double beta = 0.001;
    double threshold = 1e-6;
    int MAX_ITER = 10000000;
    int SUB_MAX_ITER = 10000;
    Condat_Vu condat_vu(F, G, H, A, dim1, dim2, alpha, beta, beta, threshold, MAX_ITER, SUB_MAX_ITER);
    PD3O pd3o(F, G, H, A, dim1, dim2, alpha, beta, beta, threshold, MAX_ITER, SUB_MAX_ITER);

    condat_vu.solve(init1, init2);
    pd3o.solve(init1, init2);
}