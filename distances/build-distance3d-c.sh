#!/bin/bash

# Invoke GNU C compiler to produce a source listing with assembly code
# and an object file, but not a linked executable file.
gcc -S -fverbose-asm -g -o distance_3d.s distance_3d.c

# Invoke c++filt to read the source listing and produce a more readable
# assembler listing with C statements paired to corresponding instructions.
c++filt < distance_3d.s | as -alhnd > distance_3d.lst

# Invoke GNU C compiler to build an executable program with debug symbols.
gcc -g -o distance_3d distance_3d.c -lm
