#!/bin/bash

# Invoke GNU C compiler to produce a source listing with assembly code
# and an object file, but not a linked executable file.
gcc -S -fverbose-asm -g -o perceptron_xor.s perceptron_xor.c

# Invoke c++filt to read the source listing and produce a more readable
# assembler listing with C statements paired to corresponding instructions.
c++filt < perceptron_xor.s | as -alhnd > perceptron_xor.lst

# Invoke GNU C compiler to build an executable program with debug symbols.
gcc -g -o perceptron_xor perceptron_xor.c -lm
