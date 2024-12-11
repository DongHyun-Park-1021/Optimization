cmake_minimum_required(VERSION 3.30)
project(Optimization)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)

set(SDL2_PATH "/Users/donghyunpark/Desktop/SDL/SDL2.framework")

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIR})

add_executable(Optimization main.cpp
        FPI.h
        Prox.h
        PGM.h
        Derivative.h
        DRS_FPI.h
        Func.h
        Lasso.h
        Matrix.h
)

target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARY})
