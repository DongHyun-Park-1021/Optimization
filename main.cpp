#include <iostream>
#include <vector>
#include <cstdlib>
#include <SDL.h>
#include <fstream>
#include <sstream>

#include "Matrix.h"
#include "LASSO.h"

#define ROW 200
#define COL 11

const int windowHeight = 600;
const int windowWidth = 800;

using namespace std;

void loadData(Matrix* A, vector<double>* b) {
    ifstream file("/Users/donghyunpark/CLionProjects/Optimization/winequality-red.csv");
    if (!file.is_open()) {
        cout << "Error opening file" << endl;
        return;
    }

    string data[ROW + 1][COL + 1];
    string line;
    int row = 0;
    while (getline(file, line) && row < ROW + 1) {
        stringstream ss(line);
        string cell;
        int col = 0;
        while (getline(ss, cell, ';') && col < COL + 1) {
            data[row][col] = cell;
            col++;
        }
        row++;
    }
    file.close();

    for (int i = 1; i < ROW + 1; i++) {
        for (int j = 0; j < COL && !data[i][j].empty(); j++) {
            ((*A).matrix)[i - 1][j] = stod(data[i][j]);
        }
        (*b)[i - 1] = stod(data[i][COL]);
    }
}

int main() {
    Matrix A = Matrix(ROW, COL);
    vector<double> b(ROW);
    vector<double> x0(COL);
    for (int i = 0; i < COL; i++) { x0[i] = 0.0; }

    loadData(&A, &b);

    LASSO lasso1 = LASSO(ROW, COL, A, b, 0.0000001, 1, 10000000000, 0.0000000001);
    LASSO lasso2 = LASSO(ROW, COL, A, b, 0.0000001, 1, 10000000000, 0.0000000001);

    //lasso1.solve_PGM(x0);
    cout << endl;


    lasso2.solve_DRS(x0);
}
