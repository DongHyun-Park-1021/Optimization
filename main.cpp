#include <iostream>
#include <vector>
#include <functional>
#include <cstdlib>
#include <SDL.h>
#include <float.h>

#include "Proj.h"

const int windowHeight = 600;
const int windowWidth = 800;

int dim = 2;
int Resolution = 200;
double mesh_size = 0.01;

using namespace std;

double f(vector<double> x) {
    return x.at(0) * x.at(0) + 3 * x.at(1) * x.at(1) - 3;
}

vector<vector<double>> Create_Mesh() {
    vector<vector<double>> mesh;
    vector<double> p(dim);

    for (int x = (-1) * windowWidth / 2 ; x < windowWidth / 2 ; x += 5) {
        p[0] = x * mesh_size;
        for (int y = (-1) * windowHeight / 2 ; y < windowHeight / 2 ; y += 5) {
            p[1] = y * mesh_size;
            if (f(p) <= 0) {
                mesh.push_back(p);
            }
        }
    }
    return mesh;
}

void Draw(SDL_Renderer* Renderer, vector<vector<double>> mesh, vector<vector<double>> p) {
    vector<double> m;

    SDL_SetRenderDrawColor(Renderer, 255, 255, 255, 255);
    for (int i = 0; i < mesh.size(); i++) {
        m = mesh.at(i);
        SDL_RenderDrawPoint(Renderer, round(m.at(0) * Resolution) + windowWidth/2, round(m.at(1) * Resolution) + windowHeight/2);
    }

    SDL_SetRenderDrawColor(Renderer, 255, 120, 120, 255);
    for (int i = 0; i < p.size(); i++) {
        m = p.at(i);

        SDL_RenderDrawPoint(Renderer, round( m.at(0) * Resolution) + windowWidth/2, round(m.at(1) * Resolution) + windowHeight/2);
    }

    SDL_RenderPresent(Renderer);
}

int main() {
    vector<double> feasible(dim);
    feasible.at(0) = 0;
    feasible.at(1) = 0;

    vector<double> x0(dim);
    x0.at(0) = 1;
    x0.at(1) = -1;

    Proj proj = Proj(feasible, f, dim, 0.25, 0.001, 10000);
    proj.Project(x0);

    vector<vector<double>> p;
    p = proj.GetData();

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

        Draw(s, Create_Mesh(), p);
    }

    SDL_DestroyWindow(window);
    SDL_DestroyRenderer(s);
    SDL_Quit();

}