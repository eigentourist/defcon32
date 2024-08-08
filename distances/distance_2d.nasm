; distance_2d.nasm
; Compile with: nasm -f elf64 distance_2d.nasm
; Link with: ld -o distance_2d distance_2d.o -lc -lm -dynamic-linker /lib64/ld-linux-x86-64.so.2

extern printf
extern sqrt

section .data
    point1: dq 1.0, 2.0
    point2: dq 4.0, 6.00
    fmt:    db "Points: (%.2f, %.2f) and (%.2f, %.2f)", 10
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
    movsd xmm2, [point2]    ; x2
    movsd xmm3, [point2+8]  ; y2

    ; Calculate distance
    movsd xmm4, xmm2
    subsd xmm4, xmm0        ; dx = x2 - x1
    movsd xmm5, xmm3
    subsd xmm5, xmm1        ; dy = y2 - y1
    mulsd xmm4, xmm4        ; dx^2
    mulsd xmm5, xmm5        ; dy^2
    addsd xmm5, xmm4        ; dy^2 + dx^2
    sqrtsd xmm4, xmm5       ; sqrt(dy^2 + dx^2)

    ; Prepare for printf
    mov rdi, fmt
    ; xmm0-xmm3 already contain the point coordinates
    ; xmm5 contains the distance
    mov rax, 6  ; using 6 XMM registers (0-5)
    call printf

    leave
    ret