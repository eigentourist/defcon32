#!/bin/bash

# Invoke Netwide Assembler, link math library for square root function.
# About the -dynamic-linker option:
# This option tells the linker to write the path of the dynamic linker into the executable.
# When the OS loads this program, it loads whatever program it finds on that path to handle dynamic linking.
# If we don't specify the dynamic linker, the program won't know how to load any shared libraries like libc (for printf) or libm (for sqrt).
# Not being able to find the libraries it needs to link in once it starts running, the program will crash.

nasm -f elf64 distance_3d.nasm
ld -o distance_3d distance_3d.o -lc -lm -dynamic-linker /lib64/ld-linux-x86-64.so.2
