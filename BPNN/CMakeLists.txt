cmake_minimum_required(VERSION 3.10)
project(BPNN)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_EXE_LINKER_FLAGS -static)

add_executable(BPNN main.cpp BPnet.h BPnet.cpp read.cpp)