; distance_2d.nasm
; Compile with: nasm -f elf64 distance_2d.nasm
; Link with: ld -o distance_2d distance_2d.o -lc -lm -dynamic-linker /lib64/ld-linux-x86-64.so.2

extern printf
extern sqrt

section .data
    point1: dq 1.0, 2.0, 3.0
    point2: dq 4.0, 6.0, 8.0
    fmt:    db "Points: (%.2f, %.2f, %.2f) and (%.2f, %.2f, %.2f)", 10
            db "Distance: %.2f", 10, 0

section .text
    global _start

_start:
    ; Call main function
    call main

    ; Exit the program
    mov rax, 60         ; sys_exit
    xor rdi, rdi        ; status: 0
    syscall

main:
    push rbp
    mov rbp, rsp

    ; Load points
    movsd xmm0, [point1]    ; x1
    movsd xmm1, [point1+8]  ; y1
    movsd xmm2, [point1+16] ; z1
    movsd xmm3, [point2]    ; x2
    movsd xmm4, [point2+8]  ; y2
    movsd xmm5, [point2+16] ; z2


    ; Calculate distance
    movsd xmm6, xmm3
    subsd xmm6, xmm0        ; dx = x2 - x1
    movsd xmm7, xmm4
    subsd xmm7, xmm1        ; dy = y2 - y1
    movsd xmm8, xmm5
    subsd xmm8, xmm2        ; dz = z2 - z1
    mulsd xmm6, xmm6        ; dx^2
    mulsd xmm7, xmm7        ; dy^2
    mulsd xmm8, xmm8        ; dz^2
    addsd xmm6, xmm7        ; dy^2 + dx^2
    addsd xmm6, xmm8        ; + dz^2
    sqrtsd xmm6, xmm6       ; sqrt(dy^2 + dx^2)

    ; Prepare for printf
    mov rdi, fmt
    ; xmm0-xmm3 already contain the point coordinates
    ; xmm5 contains the distance
    mov rax, 7  ; using 7 XMM registers (0-6) - 0 thru 5 for coordinates and 6 for distance
    call printf

    leave
    ret