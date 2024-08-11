#!/bin/bash

# Invoke GNU C compiler to build an executable program with debug symbols.
g++ -g -std=c++11 simpleRNN.cpp temps.cpp -o temps
