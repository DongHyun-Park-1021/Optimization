#pragma once

#ifndef LASSO_H
#define LASSO_H

#define DataStore_Period 10000000

#include "cmath"
#include "Matrix.h"


using namespace std;

class LASSO {
public:
    unsigned int R, C;
    Matrix A = Matrix();
    vector<double> b;
    double alpha, lambda;
    int MAX_ITER;
    double threshold;

    vector<vector<double>> data;

    LASSO(unsigned int _R, unsigned int _C, Matrix _A, vector<double> _b, double _alpha, double _lambda, int MAX_ITER, double threshold);
    void solve_PGM(vector<double> x0);
    void solve_DRS(vector<double> z0);

    vector<vector<double>> Get_Data();
};

inline LASSO::LASSO(unsigned int _R, unsigned int _C, Matrix _A, vector<double> _b, double _alpha, double _lambda, int _MAX_ITER, double _threshold) {
    R = _R;
    C = _C;
    A = _A;
    b = _b;
    alpha = _alpha;
    lambda = _lambda;
    MAX_ITER = _MAX_ITER;
    threshold = _threshold;
}

inline vector<vector<double>> LASSO::Get_Data() {
    return data;
}

inline void LASSO::solve_PGM(vector<double> x0) {
    vector<double> x = x0, y(R), z(C);
    vector<double> prev = x0;
    double rel_diff = 0;
    data.push_back(x0);
    Matrix At = A.Transpose() * alpha;
    double bd = alpha * lambda;
    const clock_t begin = clock();

    for (int i = 0 ; i < MAX_ITER ; i++) {
        y = A * x;
        for (int j = 0 ; j < R ; j++) {
            y.at(j) -= b.at(j);
        }
        z = At * y;
        for (int j = 0 ; j < C ; j++) {
            x.at(j) -= z.at(j);
            if (x.at(j) < bd && x.at(j) > (-1) * bd) x.at(j) = 0;
            else if (x.at(j) > bd) x.at(j) -= bd;
            else x.at(j) += bd;
        }

        rel_diff = 0;
        for (int j = 0 ; j < C ; j++) {
            rel_diff += ((x.at(j) - prev.at(j)) * (x.at(j) - prev.at(j)));
        }

        if (i % DataStore_Period == 0) { data.push_back(x); }

        if (rel_diff < threshold * threshold) {
            cout << "LASSO by PGM Method converged with iterate : " << i << endl;
            cout << "LASSO by PGM Method converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
            for (int j = 0 ;  j < C ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
            return;
        }

        prev = x;
    }
    cout << "LASSO by PGM Method not converged" << endl;
    cout << "LASSO by PGM Method time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

inline void LASSO::solve_DRS(vector<double> z0) {
    vector<double> z = z0;
    vector<double> prev(C), x(C), x_half(C), y(C);
    double rel_diff = 0;

    Matrix D = Matrix(C, C);
    D = alpha * A.Transpose() * A;
    D = D.Special_Inverse();

    vector<double> alpha_At_b = (alpha * A.Transpose()) * b;
    double bd = alpha * lambda;
    const clock_t begin = clock();

    for (int i = 0 ;  i < MAX_ITER ; i++) {
        for (int j = 0 ; j < C ; j++) {
            x_half.at(j) = z.at(j) + alpha_At_b.at(j);
        }
        x_half = D * x_half;

        for (int j = 0 ; j < C ; j++) {
            x.at(j) = 2 * x_half.at(j) - z.at(j);
            if (x.at(j) < bd && x.at(j) > (-1) * bd) x.at(j) = 0;
            else if (x.at(j) > bd) x.at(j) -= bd;
            else x.at(j) += bd;

            z.at(j) += x.at(j) - x_half.at(j);
        }

        if (i % DataStore_Period == 0) { data.push_back(x); }

        if (i == 0) {
            cout << "Initial x : ";
            for (int j = 0 ; j < C ; j++) {
                cout << x.at(j) << " ";
            }
            cout << endl;
        }

        if (i > 0) {
            rel_diff = 0;
            for (int j = 0 ; j < C ; j++) {
                rel_diff += (x.at(j) - prev.at(j)) * (x.at(j) - prev.at(j));
            }
            if (rel_diff < threshold * threshold) {
                cout << "LASSO by DRS Method converged with iterate : " << i << endl;
                cout << "LASSO by DRS Method converged with time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
                for (int j = 0 ;  j < C ; j++) {
                    cout << x.at(j) << " ";
                }
                cout << endl;
                return;
            }
        }

        prev = x;
    }
    cout << "LASSO by DRS Method not converged" << endl;
    cout << "LASSO by DRS Method time : " << float(clock() - begin) / CLOCKS_PER_SEC << " sec" << endl;
}

#endif //LASSO_H
