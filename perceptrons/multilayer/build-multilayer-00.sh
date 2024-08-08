#!/bin/bash

# Invoke GNU C compiler to produce a source listing with assembly code
# and an object file, but not a linked executable file.
#g++ -std=c++17 -S -fverbose-asm -g -o multilayer_perceptron.s multilayer_perceptron.cpp

# Invoke c++filt to read the source listing and produce a more readable
# assembler listing with C statements paired to corresponding instructions.
#c++filt < multilayer_perceptron.s | as -alhnd > multilayer_perceptron.lst

# Invoke GNU C compiler to build an executable program with debug symbols.
g++ -std=c++11 multilayer_00.cpp -o multilayer_00