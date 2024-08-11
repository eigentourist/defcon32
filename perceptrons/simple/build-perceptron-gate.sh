#!/bin/bash

# Invoke GNU C compiler to produce a source listing with assembly code
# and an object file, but not a linked executable file.
gcc -S -fverbose-asm -g -o perceptron_gate.s perceptron_gate.c

# Invoke c++filt to read the source listing and produce a more readable
# assembler listing with C statements paired to corresponding instructions.
c++filt < perceptron_gate.s | as -alhnd > perceptron_gate.lst

# Invoke GNU C compiler to build an executable program with debug symbols.
gcc -g -o perceptron_gate perceptron_gate.c -lm
