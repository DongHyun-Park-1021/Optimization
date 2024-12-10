#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <SDL.h>
#include <float.h>

#include "FPI.h"
#include "DRS_FPI.h"

const int windowHeight = 600;
const int windowWidth = 800;

int dim = 2;
int Resolution = 200;
double mesh_size = 0.005;

using namespace std;

double f(vector<double> x) {
    return (x.at(0) - 1) * (x.at(0) - 1) + (x.at(1) - 0.5) * (x.at(1) - 0.5) * (x.at(1) - 0.5) * (x.at(1) - 0.5) - 0.5;
}

double g(vector<double> x) {
    return x.at(0) * x.at(0) + 3 * x.at(1) * x.at(1) - 3;
}

vector<vector<double>> Create_Mesh(std::function<double(vector<double>)> func) {
    vector<vector<double>> mesh;
    vector<double> p(dim);

    for (int x = (-1) * windowWidth / 2 ; x < windowWidth / 2 ; x += 5) {
        p[0] = x * mesh_size;
        for (int y = (-1) * windowHeight / 2 ; y < windowHeight / 2 ; y += 5) {
            p[1] = y * mesh_size;
            if (func(p) <= 0) {
                mesh.push_back(p);
            }
        }
    }
    return mesh;
}

vector<vector<double>> Create_joint_Mesh() {
    vector<vector<double>> mesh;
    vector<double> p(dim);

    for (int x = (-1) * windowWidth / 2 ; x < windowWidth / 2 ; x += 5) {
        p[0] = x * mesh_size;
        for (int y = (-1) * windowHeight / 2 ; y < windowHeight / 2 ; y += 5) {
            p[1] = y * mesh_size;
            if (f(p) <= 0 && g(p) <= 0) {
                mesh.push_back(p);
            }
        }
    }
    return mesh;
}

void Draw(SDL_Renderer* Renderer, vector<vector<double>> mesh1, vector<vector<double>> mesh2, vector<vector<double>> joint_mesh, vector<vector<double>> p) {
    vector<double> m;

    SDL_SetRenderDrawColor(Renderer, 64, 64, 128, 255);
    for (int i = 0; i < mesh1.size(); i++) {
        m = mesh1.at(i);

        SDL_RenderDrawPoint(Renderer, round( ( m.at(0)) * Resolution) + windowWidth/2, round( (m.at(1) ) * Resolution) + windowHeight/2);
    }

    SDL_SetRenderDrawColor(Renderer, 64, 128, 64, 255);
    for (int i = 0; i < mesh2.size(); i++) {
        m = mesh2.at(i);

        SDL_RenderDrawPoint(Renderer, round( ( m.at(0)) * Resolution) + windowWidth/2, round( (m.at(1) ) * Resolution) + windowHeight/2);
    }

    SDL_SetRenderDrawColor(Renderer, 180, 180, 180, 255);
    for (int i = 0; i < joint_mesh.size(); i++) {
        m = joint_mesh.at(i);

        SDL_RenderDrawPoint(Renderer, round( ( m.at(0)) * Resolution) + windowWidth/2, round( (m.at(1) ) * Resolution) + windowHeight/2);
    }

    SDL_SetRenderDrawColor(Renderer, 255, 120, 120, 255);
    for (int i = 0; i < p.size(); i+=4) {
        m = p.at(i);

        SDL_RenderDrawPoint(Renderer, round( (m.at(0)) * Resolution) + windowWidth/2, round( (m.at(1) ) * Resolution) + windowHeight/2);
    }

    SDL_RenderPresent(Renderer);
}

int main() {
    vector<double> start(2);
    start.at(0) = -0.375953;
    start.at(1) = +1.08393;

    vector<double> feasible1(2), feasible2(2);
    feasible1.at(0) = 0;
    feasible1.at(1) = 0;
    feasible2.at(0) = 1;
    feasible2.at(1) = 0.5;

    Func F = Func(f, dim, true, feasible2);
    Func G = Func(g, dim, true, feasible1);


    start.at(0) = 3.5;
    start.at(1) = 2;

    DRS_FPI drs = DRS_FPI(F, G, 2, 0.005, 0.005, 0.0001, 1000, 500);
    vector<vector<double>> ret_drs;
    drs.solve(start);
    ret_drs = drs.GetData();


    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        return 1;
        cout << "Initialization failed" << endl;
    }

    SDL_Window *window = SDL_CreateWindow("Practice making sdl Window",
            SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, windowWidth,
            windowHeight, SDL_WINDOW_SHOWN);

    if (window == NULL) {
        SDL_Quit();
        return 2;
    }

    // We create a renderer with hardware acceleration, we also present according with the vertical sync refresh.
    SDL_Renderer *s = SDL_CreateRenderer(window, 0, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC) ;

    bool quit = false;
    SDL_Event event;
    while (!quit) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                quit = true;
            }
        }
        SDL_SetRenderDrawColor(s, 0, 0, 0, 255);
        SDL_RenderClear(s);

        Draw(s, Create_Mesh(f), Create_Mesh(g), Create_joint_Mesh(), ret_drs);
    }

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(s);
    SDL_Quit();

}
