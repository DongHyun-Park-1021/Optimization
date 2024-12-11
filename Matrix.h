#pragma once

#ifndef MATRIX_H
#define MATRIX_H

#define INV_CAL_MAX 50

#include <iostream>
#include <ostream>
#include <vector>


using namespace std;

class Matrix {
public:
    unsigned int R, C;
    vector<vector<double>> matrix;

    Matrix();
    Matrix(unsigned int _R, unsigned int _C);
    Matrix(unsigned int _R, unsigned int _C, vector<vector<double>> _matrix);

    friend ostream& operator<<(ostream& os, const Matrix& matrix);

    void Identity();
    Matrix Transpose();
    Matrix Special_Inverse(); // We will solve inverse problem (I+A)^{-1}
};

inline Matrix::Matrix() {}

inline Matrix::Matrix(unsigned int _R, unsigned int _C) {
    R = _R;
    C = _C;
    vector<double> tmp(C);
    for (unsigned int i = 0; i < C; i++) {tmp.at(i) = 0.0;}
    for (int i = 0; i < R; i++) { matrix.push_back(tmp); }
}

inline Matrix::Matrix(unsigned int _R, unsigned int _C, vector<vector<double>> _matrix) {
    R = _R;
    C = _C;
    matrix = _matrix;
}

ostream& operator<<(ostream& os, const Matrix& matrix) {
    for (int i = 0; i < matrix.R; i++) {
        for (int j = 0; j < matrix.C; j++) {
            os << matrix.matrix[i][j] << " ";
        }
        os << endl;
    }
    return os;
}

Matrix operator+(const Matrix& m1, const Matrix& m2) {
    if (m1.R != m2.R || m1.C != m2.C) {
        cout << "Matrix Addition failed by dimension error!" << endl;
        return Matrix(0,0);
    }
    Matrix res = Matrix(m1.R, m1.C);
    for (int i = 0 ; i < m1.R; i++) {
        for (int j = 0 ; j < m1.C; j++) {
            res.matrix[i][j] = m1.matrix[i][j] + m2.matrix[i][j];
        }
    }
    return res;
}

Matrix operator-(const Matrix& m1, const Matrix& m2) {
    if (m1.R != m2.R || m1.C != m2.C) {
        cout << "Matrix Addition failed by dimension error!" << endl;
        return Matrix(0,0);
    }
    Matrix res = Matrix(m1.R, m1.C);
    for (int i = 0 ; i < m1.R; i++) {
        for (int j = 0 ; j < m1.C; j++) {
            res.matrix[i][j] = m1.matrix[i][j] - m2.matrix[i][j];
        }
    }
    return res;
}

Matrix operator*(const Matrix& m, const double alpha) {
    Matrix res = Matrix(m.R, m.C);
    for (int i = 0 ; i < m.R; i++) {
        for (int j = 0 ; j < m.C; j++) {
            res.matrix[i][j] = m.matrix[i][j] * alpha;
        }
    }
    return res;
}
Matrix operator*(const double alpha, const Matrix& m) {
    Matrix res = Matrix(m.R, m.C);
    for (int i = 0 ; i < m.R; i++) {
        for (int j = 0 ; j < m.C; j++) {
            res.matrix[i][j] = m.matrix[i][j] * alpha;
        }
    }
    return res;
}

Matrix operator/(const Matrix& m, const double alpha) {
    Matrix res = Matrix(m.R, m.C);
    for (int i = 0 ; i < m.R; i++) {
        for (int j = 0 ; j < m.C; j++) {
            res.matrix[i][j] = m.matrix[i][j] / alpha;
        }
    }
    return res;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
    if (m1.C != m2.R) {
        cout << "Matrix multiplication failed by dimension error!" << endl;
        return Matrix(0,0);
    }
    Matrix res = Matrix(m1.R, m2.C);
    for (int i = 0 ; i < m1.R; i++) {
        for (int k = 0 ; k < m2.C; k++) {
            for (int j = 0 ; j < m1.C; j++) {
                res.matrix[i][k] += m1.matrix[i][j] * m2.matrix[j][k];
            }
        }
    }
    return res;
}

vector<double> operator*(const Matrix& m, const vector<double> &v) {
    if (m.C != v.size()) {
        cout << "Matrix vector multiplication failed by dimension error!" << endl;
        return v;
    }
    vector<double> res(m.R);
    for (int i = 0 ; i < m.R; i++) {
        res[i] = 0.0;
        for (int j = 0 ; j < m.C; j++) {
            res[i] += m.matrix[i][j] * v[j];
        }
    }
    return res;
}

inline void Matrix::Identity () {
    if (R != C) {
        cout << "Matrix identity failed by dimension error!" << endl;
        return;
    }
    for (int i = 0 ; i < R ; i++) {
        for (int j = 0 ; j < C ; j++) {
            matrix[i][j] = 0.0;
        }
        matrix[i][i] = 1.0;
    }
}

inline Matrix Matrix::Transpose() {
    Matrix res = Matrix(C, R);
    for (int i = 0 ; i < R; i++) {
        for (int j = 0 ; j < C; j++) {
            res.matrix[j][i] = matrix[i][j];
        }
    }
    return res;
}

inline Matrix Matrix::Special_Inverse() {
    if (R != C) {
        cout << "Matrix inverse failed by dimension error!" << endl;
    }

    Matrix res = Matrix(R, C);
    res.Identity();

    Matrix mul = (*this);

    for (int i = 0 ; i < INV_CAL_MAX ; i++) {
        if (i % 2 == 0) {
            res = res - mul;
        }else {
            res = res + mul;
        }
        mul = mul * (*this);
    }
    return res;
}

#endif //MATRIX_H
