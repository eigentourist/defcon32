#!/bin/bash

# Invoke GNU C compiler to produce a source listing with assembly code
# and an object file, but not a linked executable file.
gcc -S -fverbose-asm -g -o perceptron_01.s perceptron_01.c

# Invoke c++filt to read the source listing and produce a more readable
# assembler listing with C statements paired to corresponding instructions.
c++filt < perceptron_01.s | as -alhnd > perceptron_01.lst

# Invoke GNU C compiler to build an executable program with debug symbols.
gcc -g -o perceptron_01 perceptron_01.c -lm
