cmake_minimum_required(VERSION 3.10)
project(MyNativeBayes)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXE_LINKER_FLAGS -static)

add_executable(MyNativeBayes main.cpp read.cpp Matrix.h BayesClassifier.cpp BayesClassifier.h)