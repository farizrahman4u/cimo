/* Every instruction form, operand form and directive cimo.backend's amd64 emitters can produce
   (QBE amd64/emit.c: omap[], emitins, emitf, amd64_sysv_emitfn, amd64_winabi_emitfn; the data
   emitter python/cimo/backend/emit.py), plus the extras cimo.assembler_amd64 accepts, in ELF
   syntax. test_assembler_amd64.py assembles this with GNU as and with cimo.assembler_amd64 and
   compares the objects. */
.text
.balign 16
.globl forms
forms:
	endbr64
	/* prologues and epilogues (sysv and win) */
	pushq %rbp
	movq %rsp, %rbp
	subq $24, %rsp
	subq $4096, %rsp
	subq $-32, %rsp
	movq %rdi, -176(%rbp)
	movq %r9, -136(%rbp)
	movaps %xmm0, -128(%rbp)
	movaps %xmm7, -16(%rbp)
	movq %rcx, 0x8(%rsp)
	movq %r9, 0x20(%rsp)
	pushq %rbx
	pushq %r12
	pushq %r15
	popq %r15
	popq %r12
	popq %rbx
	movq %rbp, %rsp
	addq $24, %rsp
	addq $4096, %rsp
	leave
	leaveq
	ret
	retq
	ud2
	/* integer ops, both widths, register, immediate and memory operands */
	addl %eax, %ecx
	addq %r8, %r15
	addl $1, %r9d
	addq $127, %rax
	addq $128, %rax
	addl $-128, %ebx
	addl $-129, %ebx
	addq $2147483647, %rsp
	addq $-2147483648, %r13
	addl $4294967295, %eax
	addl $0xffffff80, %edx
	addl -8(%rbp), %eax
	addq %rax, -8(%rbp)
	addl $7, (%rax)
	addq $1000, 8(%rax,%rcx,8)
	subl %eax, %ecx
	subq $1, %rax
	subq $300, %rdi
	subq 16(%rsp), %r11
	andl $255, %eax
	andq $-16, %rsp
	andl %r10d, %r11d
	andq 8(%r12), %r13
	orl $1, %eax
	orq %rcx, %rdx
	orq $0x7f000000, %rcx
	xorl %eax, %eax
	xorq %r8, %r8
	xorl $-1, %esi
	cmpl %eax, %ecx
	cmpq $0, %rax
	cmpl $200, %eax
	cmpq $-1, 8(%rbp)
	cmpl $1000, -4(%rbp)
	cmpq (%rdi), %rsi
	cmpq %rsi, (%rdi)
	adcl $1, %eax
	adcq %rax, %rbx
	sbbl %eax, %eax
	sbbq $100000, %r8
	testl %eax, %eax
	testq %rcx, %rdx
	testl %eax, -8(%rbp)
	testq -8(%rbp), %rax
	testl $1, %eax
	testl $1, %ecx
	testq $256, %rdi
	testb $1, %al
	testb $1, %cl
	testb $1, %sil
	testw $1, %ax
	testw $1, %bx
	testb %al, %bl
	testb $8, (%rax)
	/* accumulator short forms and byte/word sizes */
	addb $5, %al
	addb $5, %bl
	addb $5, %dil
	addb $-1, %r8b
	addw $5, %ax
	addw $500, %ax
	addw $500, %cx
	addw %ax, %cx
	addl $500, %eax
	addq $500, %rax
	addq $-500, %rax
	cmpb $10, %al
	cmpb $10, (%rdi)
	cmpw $1000, %ax
	cmpw $-2, 6(%rsp)
	orb %al, %cl
	andb %sil, %dil
	xorb (%rax), %r9b
	subw (%rax), %r10w
	/* multiplication and division */
	imull %ecx, %eax
	imulq %r8, %rax
	imulq -8(%rbp), %rcx
	imull $3, %eax, %ecx
	imulq $100, %rax, %rax
	imulq $-100000, %rdx, %r9
	imull $1000, -4(%rbp), %r8d
	imull $10, %eax
	imulq $1000, %rcx
	imulw $3, %ax, %cx
	imulw %ax, %bx
	imull %eax
	imulq (%rax)
	cltd
	cqto
	cltq
	cwtl
	cbtw
	cwtd
	cdq
	cqo
	cdqe
	cwde
	idivl %ecx
	idivq %r8
	idivl -4(%rbp)
	divl %ecx
	divq (%rsi)
	mull %ecx
	mulq %r11
	negl %eax
	negq %r12
	negl -4(%rbp)
	notl %eax
	notq %r13
	notb %al
	incl %eax
	incq (%rax)
	decq %r9
	decb %cl
	/* shifts */
	sarl %cl, %eax
	sarq %cl, %r9
	shrl %cl, %eax
	shrq %cl, %rdx
	shll %cl, %esi
	shlq %cl, -8(%rbp)
	sarl $1, %eax
	shrq $1, %rax
	shll $1, (%rax)
	shll $2, %eax
	shlq $63, %r10
	shrq $32, %rax
	sarq $5, -8(%rbp)
	sall $3, %ecx
	salq %cl, %rax
	rolq $3, %rax
	rorl %cl, %ebx
	rolw $8, %ax
	rorl $1, %r9d
	shrb $1, %al
	sarb %cl, %r8b
	shlw $4, %cx
	shrl %eax
	shlq %rdx
	/* moves */
	movl %eax, %ecx
	movq %rax, %rcx
	movl %eax, %eax
	movl %r8d, %r8d
	movq %r15, %rax
	movb %al, %cl
	movb %dl, (%rcx)
	movb %sil, (%rdi)
	movb %r9b, -1(%rbp)
	movw %ax, (%rdx)
	movw %r8w, 2(%rsp)
	movl %eax, 4(%rsp)
	movq %rax, (%rsp)
	movq %r12, (%r12)
	movq %r13, (%r13)
	movq %rbp, 0(%rbp)
	movq %rax, (%rax, %r8, 1)
	movq %rax, (%rax,%r13,2)
	movq %rax, 128(%r12,%rbp,4)
	movq %rax, -129(%r13,%r14,8)
	movb $0, (%rax, %r8, 1)
	movq (,%rax,8), %rcx
	movl 16(,%rcx,4), %edx
	movq 8(%rax), %rcx
	movl -4(%rbp), %eax
	movw -2(%rbp), %ax
	movb -1(%rbp), %al
	movl $0, %eax
	movl $1, %r8d
	movl $-1, %eax
	movl $4294967295, %ecx
	movl $81985529216486895, -24(%rbp)
	movl $81985529216486895>>32, 4+-24(%rbp)
	movl $81985529216486895, %ecx
	movq $0, %rax
	movq $-1, %rax
	movq $2147483647, %r9
	movq $-2147483648, %r9
	movq $2147483648, %r9
	movq $4294967295, %rax
	movq $81985529216486895, %r15
	movq $-81985529216486895, %rax
	movabsq $1, %rax
	movabsq $81985529216486895, %r10
	movq $9, 72(%rax)
	movq $-7, 72(%rax)
	movl $5, (%rdi)
	movw $1000, (%rdi)
	movw $-1, %ax
	movw $65535, %r8w
	movb $255, %al
	movb $-1, %r10b
	movb $65, (%rcx)
	mov %eax, %ecx
	mov %rax, %rcx
	mov -8(%rbp), %rax
	mov %ecx, -4(%rbp)
	mov $5, %eax
	/* extensions and loads of narrow values */
	movslq %eax, %rcx
	movslq -4(%rbp), %rax
	movslq %r9d, %r10
	movsbl %al, %eax
	movsbl %sil, %ecx
	movsbl (%rax), %eax
	movsbq %dil, %rax
	movsbq -1(%rbp), %r8
	movsbw %al, %cx
	movswl %ax, %eax
	movswl (%rdi), %ecx
	movswq %r9w, %r9
	movswq 2(%rax), %rax
	movzbl %al, %eax
	movzbl %bpl, %ebp
	movzbl %r12b, %r12d
	movzbl -68(%rbp), %ecx
	movzbq %al, %rax
	movzbq (%rax), %rcx
	movzbw %cl, %ax
	movzwl %ax, %eax
	movzwl (%rax), %ecx
	movzwq %cx, %rcx
	movzwq -2(%rbp), %r11
	movsx %al, %ecx
	movzx %ax, %r8
	movsxd %eax, %rcx
	/* addresses */
	leaq -88(%rbp), %rcx
	leaq (%rax,%rcx), %rdx
	leaq 8(,%rax,4), %rdx
	leaq (%rax,%rax,2), %rax
	leal 1(%rdi), %eax
	leal (%rdi,%rsi), %eax
	leaq data(%rip), %rax
	leaq data+8(%rip), %rcx
	leaq data-8(%rip), %rcx
	leaq .Ldata(%rip), %rdx
	leaq "quoted sym"(%rip), %rdx
	leaq ext(%rip), %rsi
	leaq ext+16(%rip), %rsi
	leaq forms(%rip), %rax
	leaq .Lforms_local(%rip), %rax
	leaq .Lforms_local+4(%rip), %rax
	movq ext@gotpcrel(%rip), %rax
	movq ext@GOTPCREL(%rip), %r12
	movl data(%rip), %eax
	movq %rax, data+8(%rip)
	movl $5, data(%rip)
	movq $5, data(%rip)
	movw $5, data(%rip)
	movb $5, data(%rip)
	addq $1000, data(%rip)
	cmpb $0, data(%rip)
	movl data(%rax), %eax
	movl data+4(,%rcx,4), %eax
	leaq data, %rax
	movl $data, %eax
	movq $data, %rax
	movq $data+8, %rax
	movabsq $data, %rax
	movabsq $ext+8, %rcx
	movq $ext, 8(%rsp)
	movl $ext, (%rsp)
	addq $ext, %rax
	addq $ext, %rcx
	addl $ext, %ecx
	pushq $ext
	pushq $5
	pushq $500
	pushq -8(%rbp)
	popq (%rax)
	pushq %rax
	popq %r8
	pushw %ax
	movq %fs:0, %rax
	movl %gs:0x30, %eax
	movq %fs:8(%rax), %rcx
	movq 0x1000, %rax
	movl %eax, 0x10
	/* exchanges and atomics */
	xchgq %rax, %rcx
	xchgq %rcx, %rax
	xchgq %rcx, %rdx
	xchgl %eax, %ecx
	xchgl %eax, %eax
	xchgq %rax, %rax
	xchgw %ax, %dx
	xchgl %r8d, %eax
	xchgq %r8, %r9
	xchgb %al, %cl
	xchgq %rax, (%rdi)
	xchgl (%rdi), %ecx
	lock cmpxchgq %rcx, (%rdi)
	lock cmpxchgl %ecx, 8(%rdi)
	lock cmpxchgw %cx, (%rdi)
	lock cmpxchgb %cl, (%rdi)
	cmpxchgq %r8, (%r9)
	lock xaddq %rax, (%rdi)
	lock xaddl %eax, (%rdi)
	xaddq %rax, %rcx
	lock addq $1, (%rdi)
	lock incl (%rax)
	lock orl $0, (%rsp)
	lock addl $1, %fs:(%rax)
	lock; addl $1, (%rax)
	mfence
	lfence
	sfence
	pause
	/* bit operations */
	bswapl %eax
	bswapq %rax
	bswapl %r9d
	bswapq %r15
	bsfl %eax, %ecx
	bsfq %rax, %rcx
	bsrl %eax, %ecx
	bsrq (%rax), %r8
	lzcntl %eax, %ecx
	lzcntq %rax, %rcx
	lzcntw %ax, %cx
	tzcntl %eax, %ecx
	tzcntq %r8, %r9
	popcntl %eax, %ecx
	popcntq %rax, %rcx
	popcntq 8(%rdi), %r10
	popcntw %ax, %cx
	/* flags */
	setz %al
	setnz %cl
	setb %dl
	setae %bl
	setbe %sil
	seta %dil
	setl %r8b
	setge %r9b
	setle %bpl
	setg %spl
	setp %al
	setnp %al
	sete (%rax)
	setne -1(%rbp)
	setc %al
	setnc %al
	setnae %al
	setnb %al
	setna %al
	setnbe %al
	setnge %al
	setnl %al
	setng %al
	setnle %al
	sets %al
	setns %al
	setpe %al
	setpo %al
	seto %al
	setno %al
	cmovz %eax, %ecx
	cmovz %rax, %rcx
	cmovnz %r8, %r9
	cmovb %eax, %ecx
	cmovae -8(%rbp), %rax
	cmovbe %eax, %ecx
	cmova %eax, %ecx
	cmovl %eax, %ecx
	cmovge %eax, %ecx
	cmovle %eax, %ecx
	cmovg %eax, %ecx
	cmovp %eax, %ecx
	cmovnp %eax, %ecx
	cmovs %rax, %rcx
	cmovns %rax, %rcx
	cmovo %rax, %rcx
	cmovno %rax, %rcx
	cmove %ax, %cx
	cmovne %eax, %ecx
	cmovc %eax, %ecx
	cmovnc %eax, %ecx
	cmovzl %eax, %ecx
	cmovzq %rax, %rcx
	cmovll %eax, %ecx
	cmovlq (%rax), %rcx
	cmovgel %eax, %ecx
	cmovnbeq %rax, %rcx
	/* calls and indirect jumps */
	call ext
	callq ext
	callq ext@plt
	callq ext@PLT
	callq forms
	callq .Lforms_local
	callq data
	callq ext+4
	callq *%rax
	callq *%r11
	callq *8(%rax)
	callq *data(%rip)
	call *%rax
	jmp *%rax
	jmpq *%r10
	jmp *(%rax,%rcx,8)
	jmpq *.Ltable(,%rcx,8)
	jmp ext
	jmp ext@plt
	jz ext
	jnz forms
	jmp data
	jmp ext+8
	/* misc */
	nop
	int3
	hlt
	rdtsc
	cpuid
.Lforms_local:
	/* SSE scalar float */
	movss %xmm1, %xmm0
	movss -4(%rbp), %xmm0
	movss %xmm0, -4(%rbp)
	movss %xmm8, %xmm15
	movss (%r8), %xmm9
	movss %xmm10, 4(%r11,%r12,4)
	movsd %xmm1, %xmm0
	movsd -8(%rbp), %xmm15
	movsd %xmm15, -8(%rbp)
	movsd .Lfp0(%rip), %xmm0
	movss .Lfp1(%rip), %xmm3
	movaps %xmm0, %xmm1
	movaps (%rax), %xmm9
	movaps %xmm9, (%rax)
	movapd %xmm0, %xmm1
	movapd 16(%rsp), %xmm2
	movups %xmm0, (%rdi)
	movups (%rsi), %xmm1
	movupd %xmm1, %xmm2
	movdqa %xmm0, %xmm1
	movdqa (%rax), %xmm1
	movdqa %xmm1, (%rax)
	movdqu (%rax), %xmm1
	movdqu %xmm1, (%rax)
	movq %xmm0, %rax
	movq %rax, %xmm0
	movq %xmm15, %r8
	movq %r15, %xmm8
	movq %xmm1, %xmm0
	movq %xmm9, %xmm10
	movq -8(%rbp), %xmm0
	movq %xmm0, -8(%rbp)
	movq data(%rip), %xmm3
	movd %eax, %xmm0
	movd %xmm0, %eax
	movd %r9d, %xmm12
	movd (%rax), %xmm0
	movd %xmm0, (%rax)
	movd %rax, %xmm1
	addss %xmm1, %xmm0
	addsd %xmm1, %xmm0
	addsd -8(%rbp), %xmm0
	addss .Lfp1(%rip), %xmm9
	subss %xmm1, %xmm0
	subsd %xmm8, %xmm0
	mulss %xmm1, %xmm0
	mulsd %xmm1, %xmm15
	divss %xmm1, %xmm0
	divsd 8(%rax), %xmm0
	sqrtss %xmm1, %xmm0
	sqrtsd %xmm1, %xmm0
	minss %xmm1, %xmm0
	minsd %xmm1, %xmm0
	maxss %xmm1, %xmm0
	maxsd %xmm1, %xmm0
	addps %xmm1, %xmm0
	addpd %xmm1, %xmm0
	mulps (%rax), %xmm0
	andps %xmm1, %xmm0
	andpd %xmm1, %xmm0
	andnps %xmm1, %xmm0
	andnpd %xmm1, %xmm0
	orps %xmm1, %xmm0
	orpd %xmm1, %xmm0
	xorps %xmm1, %xmm0
	xorpd %xmm1, %xmm0
	xorps .Lfp3(%rip), %xmm0
	xorpd .Lfp2(%rip), %xmm9
	pxor %xmm0, %xmm0
	pxor %xmm8, %xmm1
	pand %xmm1, %xmm2
	por %xmm1, %xmm2
	ucomiss %xmm1, %xmm0
	ucomisd %xmm1, %xmm0
	ucomisd -8(%rbp), %xmm9
	comiss %xmm1, %xmm0
	comisd %xmm1, %xmm0
	cvtss2sd %xmm1, %xmm0
	cvtss2sd -4(%rbp), %xmm0
	cvtsd2ss %xmm1, %xmm0
	cvtsd2ss %xmm8, %xmm9
	cvttss2si %xmm0, %eax
	cvttss2si %xmm0, %rax
	cvttss2sil %xmm0, %eax
	cvttss2siq %xmm0, %rax
	cvttsd2si %xmm0, %eax
	cvttsd2si %xmm0, %r8
	cvttsd2sil %xmm0, %eax
	cvttsd2siq %xmm15, %r15
	cvttsd2siq -8(%rbp), %rax
	cvtsd2si %xmm0, %rax
	cvtss2si %xmm0, %eax
	cvtsi2ss %eax, %xmm0
	cvtsi2ss %rax, %xmm0
	cvtsi2sd %eax, %xmm0
	cvtsi2sd %rax, %xmm0
	cvtsi2sd %r9, %xmm10
	cvtsi2sdl %eax, %xmm0
	cvtsi2sdq %rax, %xmm0
	cvtsi2ssl -4(%rbp), %xmm0
	cvtsi2sdq -8(%rbp), %xmm0
	cvtsi2sd -4(%rbp), %xmm0
	cvtps2pd %xmm1, %xmm0
	cvtpd2ps %xmm1, %xmm0
	unpcklps %xmm1, %xmm0
	unpcklpd %xmm1, %xmm0
	roundss $1, %xmm1, %xmm0
	roundsd $9, %xmm1, %xmm0
	roundsd $4, -8(%rbp), %xmm9
	roundps $2, %xmm1, %xmm0
	roundpd $3, %xmm1, %xmm0
	/* jumps: relaxed (rel8) and long (rel32) */
.Lbb1:
	jmp .Lbb1
	jz .Lbb1
	jnz .Lbb2
	jb .Lbb2
	jnae .Lbb2
	jc .Lbb2
	jae .Lbb2
	jnb .Lbb2
	jnc .Lbb2
	jbe .Lbb2
	jna .Lbb2
	ja .Lbb2
	jnbe .Lbb2
	jl .Lbb2
	jnge .Lbb2
	jge .Lbb2
	jnl .Lbb2
	jle .Lbb2
	jng .Lbb2
	jg .Lbb2
	jnle .Lbb2
	js .Lbb2
	jns .Lbb2
	jp .Lbb2
	jpe .Lbb2
	jnp .Lbb2
	jpo .Lbb2
	jo .Lbb2
	jno .Lbb2
	je .Lbb2
	jne .Lbb2
.Lbb2:
	jmp .Lbb3
.p2align 4
	.fill 120,1,0x90
	jle .Lbb3
	jmp forms
.Lbb3:
	jz .Lbb1
	jmp .Lbb1
	jge .Lbb4
	.fill 200,1,0x90
.p2align 4
.Lbb4:
	jmp Lbb5
	nop
Lbb5:
	jnz Lbb5
	ret
/* end function forms */

.text
.balign 16
f2:
	endbr64
	movq %rsp, %rax
	ret
.p2align 4
	ret
.balign 32
	ret
.p2align 6
	ret
.byte 0x90
.p2align 4
	ret
.type f2, @function
.size f2, .-f2
	# a comment
	ret # another

.data
.balign 8
.globl data
data:
	.byte 1
	.byte -1
	.byte 255
	.short 1000
	.short -2
	.int 100000
	.int -1
	.int 4294967295
	.long 7
	.quad 4607182418800017408
	.quad -1
	.quad 81985529216486895
	.word 1
	.value 2
	.ascii "abc\n\t\"\\\0\101\x41"
	.ascii ""
	.fill 3,1,0
	.fill 2,4,0x11
	.zero 3
.Ldata:
	.quad data
	.quad data+8
	.quad data-8
	.quad ext
	.quad ext+16
	.quad forms
	.quad .Ldata
	.int data
	.int ext+4
	.short ext
	.byte ext
	.quad "quoted sym"
.balign 8
"quoted sym":
	.quad 0
.Ltable:
	.quad .Lbb1
	.quad .Lbb2

.data
.balign 4
.globl d2
d2:
	.int 1

.bss
.balign 16
.globl zero
zero:
	.fill 100,1,0
.balign 32
zero2:
	.fill 8,1,0
.comm common1,8
.comm common2,100,32
.comm common3,3

/* floating point constants */
.section .rodata
.p2align 4
.Lfp3:
	.quad -9223372036854775808
	.quad 0

.section .rodata
.p2align 4
.Lfp2:
	.int -2147483648
	.int 0
	.int 0
	.int 0

.section .rodata
.p2align 3
.Lfp0:
	.quad 4607182418800017408 /* 1.000000 */

.section .rodata
.p2align 2
.Lfp1:
	.int 1065353216 /* 1.000000 */

.section .data.rel.ro,"aw"
.balign 8
	.quad data
.section .text.other,"ax",@progbits
	callq forms
	callq .Lforms_local
	jmp .Lforms_local
	leaq .Lforms_local(%rip), %rax
	ret
	/* AVX and AVX2 (VEX.128 and VEX.256): the 256-bit vectors on CPUs with avx2 */
	vaddpd %ymm7, %ymm1, %ymm0
	vaddpd %ymm0, %ymm15, %ymm8
	vaddpd 128(%r12,%r13,8), %ymm8, %ymm15
	vaddpd 8(%rip), %ymm0, %ymm7
	vaddpd %xmm15, %xmm9, %xmm3
	vaddpd 16(%rsp), %xmm0, %xmm15
	vaddps %ymm7, %ymm1, %ymm0
	vaddps %ymm0, %ymm15, %ymm8
	vaddps 128(%r12,%r13,8), %ymm8, %ymm15
	vaddps 8(%rip), %ymm0, %ymm7
	vaddps %xmm15, %xmm9, %xmm3
	vaddps 16(%rsp), %xmm0, %xmm15
	vandnpd %ymm7, %ymm1, %ymm0
	vandnpd %ymm0, %ymm15, %ymm8
	vandnpd 128(%r12,%r13,8), %ymm8, %ymm15
	vandnpd 8(%rip), %ymm0, %ymm7
	vandnpd %xmm15, %xmm9, %xmm3
	vandnpd 16(%rsp), %xmm0, %xmm15
	vandnps %ymm7, %ymm1, %ymm0
	vandnps %ymm0, %ymm15, %ymm8
	vandnps 128(%r12,%r13,8), %ymm8, %ymm15
	vandnps 8(%rip), %ymm0, %ymm7
	vandnps %xmm15, %xmm9, %xmm3
	vandnps 16(%rsp), %xmm0, %xmm15
	vandpd %ymm7, %ymm1, %ymm0
	vandpd %ymm0, %ymm15, %ymm8
	vandpd 128(%r12,%r13,8), %ymm8, %ymm15
	vandpd 8(%rip), %ymm0, %ymm7
	vandpd %xmm15, %xmm9, %xmm3
	vandpd 16(%rsp), %xmm0, %xmm15
	vandps %ymm7, %ymm1, %ymm0
	vandps %ymm0, %ymm15, %ymm8
	vandps 128(%r12,%r13,8), %ymm8, %ymm15
	vandps 8(%rip), %ymm0, %ymm7
	vandps %xmm15, %xmm9, %xmm3
	vandps 16(%rsp), %xmm0, %xmm15
	vblendvpd %ymm7, %ymm7, %ymm1, %ymm0
	vblendvpd %ymm8, %ymm0, %ymm15, %ymm8
	vblendvpd %ymm0, 128(%r12,%r13,8), %ymm8, %ymm15
	vblendvpd %ymm1, 8(%rip), %ymm0, %ymm7
	vblendvpd %xmm9, %xmm15, %xmm9, %xmm3
	vblendvpd %xmm15, 16(%rsp), %xmm0, %xmm15
	vblendvps %ymm7, %ymm7, %ymm1, %ymm0
	vblendvps %ymm8, %ymm0, %ymm15, %ymm8
	vblendvps %ymm0, 128(%r12,%r13,8), %ymm8, %ymm15
	vblendvps %ymm1, 8(%rip), %ymm0, %ymm7
	vblendvps %xmm9, %xmm15, %xmm9, %xmm3
	vblendvps %xmm15, 16(%rsp), %xmm0, %xmm15
	vbroadcastsd %xmm0, %ymm0
	vbroadcastsd %xmm3, %ymm8
	vbroadcastsd 128(%r12,%r13,8), %ymm15
	vbroadcastsd 8(%rip), %ymm7
	vbroadcastss %xmm0, %ymm0
	vbroadcastss %xmm3, %ymm8
	vbroadcastss 128(%r12,%r13,8), %ymm15
	vbroadcastss 8(%rip), %ymm7
	vbroadcastss %xmm0, %xmm3
	vbroadcastss 16(%rsp), %xmm15
	vcmppd $1, %ymm7, %ymm1, %ymm0
	vcmppd $2, %ymm0, %ymm15, %ymm8
	vcmppd $3, 128(%r12,%r13,8), %ymm8, %ymm15
	vcmppd $4, 8(%rip), %ymm0, %ymm7
	vcmppd $5, %xmm15, %xmm9, %xmm3
	vcmppd $6, 16(%rsp), %xmm0, %xmm15
	vcmpps $1, %ymm7, %ymm1, %ymm0
	vcmpps $2, %ymm0, %ymm15, %ymm8
	vcmpps $3, 128(%r12,%r13,8), %ymm8, %ymm15
	vcmpps $4, 8(%rip), %ymm0, %ymm7
	vcmpps $5, %xmm15, %xmm9, %xmm3
	vcmpps $6, 16(%rsp), %xmm0, %xmm15
	vdivpd %ymm7, %ymm1, %ymm0
	vdivpd %ymm0, %ymm15, %ymm8
	vdivpd 128(%r12,%r13,8), %ymm8, %ymm15
	vdivpd 8(%rip), %ymm0, %ymm7
	vdivpd %xmm15, %xmm9, %xmm3
	vdivpd 16(%rsp), %xmm0, %xmm15
	vdivps %ymm7, %ymm1, %ymm0
	vdivps %ymm0, %ymm15, %ymm8
	vdivps 128(%r12,%r13,8), %ymm8, %ymm15
	vdivps 8(%rip), %ymm0, %ymm7
	vdivps %xmm15, %xmm9, %xmm3
	vdivps 16(%rsp), %xmm0, %xmm15
	vextractf128 $1, %ymm0, %xmm0
	vextractf128 $1, %ymm1, (%rax)
	vextractf128 $1, %ymm8, %xmm3
	vextractf128 $1, %ymm15, 16(%rsp)
	vextractf128 $1, %ymm15, %xmm9
	vextractf128 $1, %ymm8, -64(%rbp)
	vextractf128 $1, %ymm7, %xmm15
	vextractf128 $1, %ymm0, 128(%r12,%r13,8)
	vextracti128 $1, %ymm0, %xmm0
	vextracti128 $1, %ymm1, (%rax)
	vextracti128 $1, %ymm8, %xmm3
	vextracti128 $1, %ymm15, 16(%rsp)
	vextracti128 $1, %ymm15, %xmm9
	vextracti128 $1, %ymm8, -64(%rbp)
	vextracti128 $1, %ymm7, %xmm15
	vextracti128 $1, %ymm0, 128(%r12,%r13,8)
	vinsertf128 $1, %xmm0, %ymm1, %ymm0
	vinsertf128 $1, %xmm3, %ymm15, %ymm8
	vinsertf128 $1, 128(%r12,%r13,8), %ymm8, %ymm15
	vinsertf128 $1, 8(%rip), %ymm0, %ymm7
	vinserti128 $1, %xmm0, %ymm1, %ymm0
	vinserti128 $1, %xmm3, %ymm15, %ymm8
	vinserti128 $1, 128(%r12,%r13,8), %ymm8, %ymm15
	vinserti128 $1, 8(%rip), %ymm0, %ymm7
	vmaxpd %ymm7, %ymm1, %ymm0
	vmaxpd %ymm0, %ymm15, %ymm8
	vmaxpd 128(%r12,%r13,8), %ymm8, %ymm15
	vmaxpd 8(%rip), %ymm0, %ymm7
	vmaxpd %xmm15, %xmm9, %xmm3
	vmaxpd 16(%rsp), %xmm0, %xmm15
	vmaxps %ymm7, %ymm1, %ymm0
	vmaxps %ymm0, %ymm15, %ymm8
	vmaxps 128(%r12,%r13,8), %ymm8, %ymm15
	vmaxps 8(%rip), %ymm0, %ymm7
	vmaxps %xmm15, %xmm9, %xmm3
	vmaxps 16(%rsp), %xmm0, %xmm15
	vminpd %ymm7, %ymm1, %ymm0
	vminpd %ymm0, %ymm15, %ymm8
	vminpd 128(%r12,%r13,8), %ymm8, %ymm15
	vminpd 8(%rip), %ymm0, %ymm7
	vminpd %xmm15, %xmm9, %xmm3
	vminpd 16(%rsp), %xmm0, %xmm15
	vminps %ymm7, %ymm1, %ymm0
	vminps %ymm0, %ymm15, %ymm8
	vminps 128(%r12,%r13,8), %ymm8, %ymm15
	vminps 8(%rip), %ymm0, %ymm7
	vminps %xmm15, %xmm9, %xmm3
	vminps 16(%rsp), %xmm0, %xmm15
	vmovapd %ymm7, %ymm0
	vmovapd %ymm0, (%rax)
	vmovapd %ymm0, %ymm8
	vmovapd %ymm8, 16(%rsp)
	vmovapd 128(%r12,%r13,8), %ymm15
	vmovapd %ymm15, -64(%rbp)
	vmovapd 8(%rip), %ymm7
	vmovapd %ymm7, 128(%r12,%r13,8)
	vmovapd %xmm15, %xmm3
	vmovapd %xmm3, 32(%r13)
	vmovapd 16(%rsp), %xmm15
	vmovapd %xmm15, 8(%rip)
	vmovaps %ymm7, %ymm0
	vmovaps %ymm0, (%rax)
	vmovaps %ymm0, %ymm8
	vmovaps %ymm8, 16(%rsp)
	vmovaps 128(%r12,%r13,8), %ymm15
	vmovaps %ymm15, -64(%rbp)
	vmovaps 8(%rip), %ymm7
	vmovaps %ymm7, 128(%r12,%r13,8)
	vmovaps %xmm15, %xmm3
	vmovaps %xmm3, 32(%r13)
	vmovaps 16(%rsp), %xmm15
	vmovaps %xmm15, 8(%rip)
	vmovdqa %ymm7, %ymm0
	vmovdqa %ymm0, (%rax)
	vmovdqa %ymm0, %ymm8
	vmovdqa %ymm8, 16(%rsp)
	vmovdqa 128(%r12,%r13,8), %ymm15
	vmovdqa %ymm15, -64(%rbp)
	vmovdqa 8(%rip), %ymm7
	vmovdqa %ymm7, 128(%r12,%r13,8)
	vmovdqa %xmm15, %xmm3
	vmovdqa %xmm3, 32(%r13)
	vmovdqa 16(%rsp), %xmm15
	vmovdqa %xmm15, 8(%rip)
	vmovdqu %ymm7, %ymm0
	vmovdqu %ymm0, (%rax)
	vmovdqu %ymm0, %ymm8
	vmovdqu %ymm8, 16(%rsp)
	vmovdqu 128(%r12,%r13,8), %ymm15
	vmovdqu %ymm15, -64(%rbp)
	vmovdqu 8(%rip), %ymm7
	vmovdqu %ymm7, 128(%r12,%r13,8)
	vmovdqu %xmm15, %xmm3
	vmovdqu %xmm3, 32(%r13)
	vmovdqu 16(%rsp), %xmm15
	vmovdqu %xmm15, 8(%rip)
	vmovmskpd %ymm0, %eax
	vmovmskpd %ymm8, %r9d
	vmovmskpd %ymm15, %ecx
	vmovmskpd %ymm7, %edx
	vmovmskpd %xmm3, %r15d
	vmovmskpd %xmm15, %esi
	vmovmskps %ymm0, %eax
	vmovmskps %ymm8, %r9d
	vmovmskps %ymm15, %ecx
	vmovmskps %ymm7, %edx
	vmovmskps %xmm3, %r15d
	vmovmskps %xmm15, %esi
	vmovupd %ymm7, %ymm0
	vmovupd %ymm0, (%rax)
	vmovupd %ymm0, %ymm8
	vmovupd %ymm8, 16(%rsp)
	vmovupd 128(%r12,%r13,8), %ymm15
	vmovupd %ymm15, -64(%rbp)
	vmovupd 8(%rip), %ymm7
	vmovupd %ymm7, 128(%r12,%r13,8)
	vmovupd %xmm15, %xmm3
	vmovupd %xmm3, 32(%r13)
	vmovupd 16(%rsp), %xmm15
	vmovupd %xmm15, 8(%rip)
	vmovups %ymm7, %ymm0
	vmovups %ymm0, (%rax)
	vmovups %ymm0, %ymm8
	vmovups %ymm8, 16(%rsp)
	vmovups 128(%r12,%r13,8), %ymm15
	vmovups %ymm15, -64(%rbp)
	vmovups 8(%rip), %ymm7
	vmovups %ymm7, 128(%r12,%r13,8)
	vmovups %xmm15, %xmm3
	vmovups %xmm3, 32(%r13)
	vmovups 16(%rsp), %xmm15
	vmovups %xmm15, 8(%rip)
	vmulpd %ymm7, %ymm1, %ymm0
	vmulpd %ymm0, %ymm15, %ymm8
	vmulpd 128(%r12,%r13,8), %ymm8, %ymm15
	vmulpd 8(%rip), %ymm0, %ymm7
	vmulpd %xmm15, %xmm9, %xmm3
	vmulpd 16(%rsp), %xmm0, %xmm15
	vmulps %ymm7, %ymm1, %ymm0
	vmulps %ymm0, %ymm15, %ymm8
	vmulps 128(%r12,%r13,8), %ymm8, %ymm15
	vmulps 8(%rip), %ymm0, %ymm7
	vmulps %xmm15, %xmm9, %xmm3
	vmulps 16(%rsp), %xmm0, %xmm15
	vorpd %ymm7, %ymm1, %ymm0
	vorpd %ymm0, %ymm15, %ymm8
	vorpd 128(%r12,%r13,8), %ymm8, %ymm15
	vorpd 8(%rip), %ymm0, %ymm7
	vorpd %xmm15, %xmm9, %xmm3
	vorpd 16(%rsp), %xmm0, %xmm15
	vorps %ymm7, %ymm1, %ymm0
	vorps %ymm0, %ymm15, %ymm8
	vorps 128(%r12,%r13,8), %ymm8, %ymm15
	vorps 8(%rip), %ymm0, %ymm7
	vorps %xmm15, %xmm9, %xmm3
	vorps 16(%rsp), %xmm0, %xmm15
	vpaddb %ymm7, %ymm1, %ymm0
	vpaddb %ymm0, %ymm15, %ymm8
	vpaddb 128(%r12,%r13,8), %ymm8, %ymm15
	vpaddb 8(%rip), %ymm0, %ymm7
	vpaddb %xmm15, %xmm9, %xmm3
	vpaddb 16(%rsp), %xmm0, %xmm15
	vpaddd %ymm7, %ymm1, %ymm0
	vpaddd %ymm0, %ymm15, %ymm8
	vpaddd 128(%r12,%r13,8), %ymm8, %ymm15
	vpaddd 8(%rip), %ymm0, %ymm7
	vpaddd %xmm15, %xmm9, %xmm3
	vpaddd 16(%rsp), %xmm0, %xmm15
	vpaddq %ymm7, %ymm1, %ymm0
	vpaddq %ymm0, %ymm15, %ymm8
	vpaddq 128(%r12,%r13,8), %ymm8, %ymm15
	vpaddq 8(%rip), %ymm0, %ymm7
	vpaddq %xmm15, %xmm9, %xmm3
	vpaddq 16(%rsp), %xmm0, %xmm15
	vpaddw %ymm7, %ymm1, %ymm0
	vpaddw %ymm0, %ymm15, %ymm8
	vpaddw 128(%r12,%r13,8), %ymm8, %ymm15
	vpaddw 8(%rip), %ymm0, %ymm7
	vpaddw %xmm15, %xmm9, %xmm3
	vpaddw 16(%rsp), %xmm0, %xmm15
	vpand %ymm7, %ymm1, %ymm0
	vpand %ymm0, %ymm15, %ymm8
	vpand 128(%r12,%r13,8), %ymm8, %ymm15
	vpand 8(%rip), %ymm0, %ymm7
	vpand %xmm15, %xmm9, %xmm3
	vpand 16(%rsp), %xmm0, %xmm15
	vpandn %ymm7, %ymm1, %ymm0
	vpandn %ymm0, %ymm15, %ymm8
	vpandn 128(%r12,%r13,8), %ymm8, %ymm15
	vpandn 8(%rip), %ymm0, %ymm7
	vpandn %xmm15, %xmm9, %xmm3
	vpandn 16(%rsp), %xmm0, %xmm15
	vpblendvb %ymm7, %ymm7, %ymm1, %ymm0
	vpblendvb %ymm8, %ymm0, %ymm15, %ymm8
	vpblendvb %ymm0, 128(%r12,%r13,8), %ymm8, %ymm15
	vpblendvb %ymm1, 8(%rip), %ymm0, %ymm7
	vpblendvb %xmm9, %xmm15, %xmm9, %xmm3
	vpblendvb %xmm15, 16(%rsp), %xmm0, %xmm15
	vpbroadcastb %xmm0, %ymm0
	vpbroadcastb %xmm3, %ymm8
	vpbroadcastb 128(%r12,%r13,8), %ymm15
	vpbroadcastb 8(%rip), %ymm7
	vpbroadcastb %xmm0, %xmm3
	vpbroadcastb 16(%rsp), %xmm15
	vpbroadcastd %xmm0, %ymm0
	vpbroadcastd %xmm3, %ymm8
	vpbroadcastd 128(%r12,%r13,8), %ymm15
	vpbroadcastd 8(%rip), %ymm7
	vpbroadcastd %xmm0, %xmm3
	vpbroadcastd 16(%rsp), %xmm15
	vpbroadcastq %xmm0, %ymm0
	vpbroadcastq %xmm3, %ymm8
	vpbroadcastq 128(%r12,%r13,8), %ymm15
	vpbroadcastq 8(%rip), %ymm7
	vpbroadcastq %xmm0, %xmm3
	vpbroadcastq 16(%rsp), %xmm15
	vpbroadcastw %xmm0, %ymm0
	vpbroadcastw %xmm3, %ymm8
	vpbroadcastw 128(%r12,%r13,8), %ymm15
	vpbroadcastw 8(%rip), %ymm7
	vpbroadcastw %xmm0, %xmm3
	vpbroadcastw 16(%rsp), %xmm15
	vpcmpeqb %ymm7, %ymm1, %ymm0
	vpcmpeqb %ymm0, %ymm15, %ymm8
	vpcmpeqb 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpeqb 8(%rip), %ymm0, %ymm7
	vpcmpeqb %xmm15, %xmm9, %xmm3
	vpcmpeqb 16(%rsp), %xmm0, %xmm15
	vpcmpeqd %ymm7, %ymm1, %ymm0
	vpcmpeqd %ymm0, %ymm15, %ymm8
	vpcmpeqd 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpeqd 8(%rip), %ymm0, %ymm7
	vpcmpeqd %xmm15, %xmm9, %xmm3
	vpcmpeqd 16(%rsp), %xmm0, %xmm15
	vpcmpeqq %ymm7, %ymm1, %ymm0
	vpcmpeqq %ymm0, %ymm15, %ymm8
	vpcmpeqq 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpeqq 8(%rip), %ymm0, %ymm7
	vpcmpeqq %xmm15, %xmm9, %xmm3
	vpcmpeqq 16(%rsp), %xmm0, %xmm15
	vpcmpeqw %ymm7, %ymm1, %ymm0
	vpcmpeqw %ymm0, %ymm15, %ymm8
	vpcmpeqw 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpeqw 8(%rip), %ymm0, %ymm7
	vpcmpeqw %xmm15, %xmm9, %xmm3
	vpcmpeqw 16(%rsp), %xmm0, %xmm15
	vpcmpgtb %ymm7, %ymm1, %ymm0
	vpcmpgtb %ymm0, %ymm15, %ymm8
	vpcmpgtb 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpgtb 8(%rip), %ymm0, %ymm7
	vpcmpgtb %xmm15, %xmm9, %xmm3
	vpcmpgtb 16(%rsp), %xmm0, %xmm15
	vpcmpgtd %ymm7, %ymm1, %ymm0
	vpcmpgtd %ymm0, %ymm15, %ymm8
	vpcmpgtd 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpgtd 8(%rip), %ymm0, %ymm7
	vpcmpgtd %xmm15, %xmm9, %xmm3
	vpcmpgtd 16(%rsp), %xmm0, %xmm15
	vpcmpgtq %ymm7, %ymm1, %ymm0
	vpcmpgtq %ymm0, %ymm15, %ymm8
	vpcmpgtq 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpgtq 8(%rip), %ymm0, %ymm7
	vpcmpgtq %xmm15, %xmm9, %xmm3
	vpcmpgtq 16(%rsp), %xmm0, %xmm15
	vpcmpgtw %ymm7, %ymm1, %ymm0
	vpcmpgtw %ymm0, %ymm15, %ymm8
	vpcmpgtw 128(%r12,%r13,8), %ymm8, %ymm15
	vpcmpgtw 8(%rip), %ymm0, %ymm7
	vpcmpgtw %xmm15, %xmm9, %xmm3
	vpcmpgtw 16(%rsp), %xmm0, %xmm15
	vpermq $5, %ymm7, %ymm0
	vpermq $42, %ymm0, %ymm8
	vpermq $79, 128(%r12,%r13,8), %ymm15
	vpermq $116, 8(%rip), %ymm7
	vpmaxsb %ymm7, %ymm1, %ymm0
	vpmaxsb %ymm0, %ymm15, %ymm8
	vpmaxsb 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxsb 8(%rip), %ymm0, %ymm7
	vpmaxsb %xmm15, %xmm9, %xmm3
	vpmaxsb 16(%rsp), %xmm0, %xmm15
	vpmaxsd %ymm7, %ymm1, %ymm0
	vpmaxsd %ymm0, %ymm15, %ymm8
	vpmaxsd 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxsd 8(%rip), %ymm0, %ymm7
	vpmaxsd %xmm15, %xmm9, %xmm3
	vpmaxsd 16(%rsp), %xmm0, %xmm15
	vpmaxsw %ymm7, %ymm1, %ymm0
	vpmaxsw %ymm0, %ymm15, %ymm8
	vpmaxsw 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxsw 8(%rip), %ymm0, %ymm7
	vpmaxsw %xmm15, %xmm9, %xmm3
	vpmaxsw 16(%rsp), %xmm0, %xmm15
	vpmaxub %ymm7, %ymm1, %ymm0
	vpmaxub %ymm0, %ymm15, %ymm8
	vpmaxub 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxub 8(%rip), %ymm0, %ymm7
	vpmaxub %xmm15, %xmm9, %xmm3
	vpmaxub 16(%rsp), %xmm0, %xmm15
	vpmaxud %ymm7, %ymm1, %ymm0
	vpmaxud %ymm0, %ymm15, %ymm8
	vpmaxud 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxud 8(%rip), %ymm0, %ymm7
	vpmaxud %xmm15, %xmm9, %xmm3
	vpmaxud 16(%rsp), %xmm0, %xmm15
	vpmaxuw %ymm7, %ymm1, %ymm0
	vpmaxuw %ymm0, %ymm15, %ymm8
	vpmaxuw 128(%r12,%r13,8), %ymm8, %ymm15
	vpmaxuw 8(%rip), %ymm0, %ymm7
	vpmaxuw %xmm15, %xmm9, %xmm3
	vpmaxuw 16(%rsp), %xmm0, %xmm15
	vpminsb %ymm7, %ymm1, %ymm0
	vpminsb %ymm0, %ymm15, %ymm8
	vpminsb 128(%r12,%r13,8), %ymm8, %ymm15
	vpminsb 8(%rip), %ymm0, %ymm7
	vpminsb %xmm15, %xmm9, %xmm3
	vpminsb 16(%rsp), %xmm0, %xmm15
	vpminsd %ymm7, %ymm1, %ymm0
	vpminsd %ymm0, %ymm15, %ymm8
	vpminsd 128(%r12,%r13,8), %ymm8, %ymm15
	vpminsd 8(%rip), %ymm0, %ymm7
	vpminsd %xmm15, %xmm9, %xmm3
	vpminsd 16(%rsp), %xmm0, %xmm15
	vpminsw %ymm7, %ymm1, %ymm0
	vpminsw %ymm0, %ymm15, %ymm8
	vpminsw 128(%r12,%r13,8), %ymm8, %ymm15
	vpminsw 8(%rip), %ymm0, %ymm7
	vpminsw %xmm15, %xmm9, %xmm3
	vpminsw 16(%rsp), %xmm0, %xmm15
	vpminub %ymm7, %ymm1, %ymm0
	vpminub %ymm0, %ymm15, %ymm8
	vpminub 128(%r12,%r13,8), %ymm8, %ymm15
	vpminub 8(%rip), %ymm0, %ymm7
	vpminub %xmm15, %xmm9, %xmm3
	vpminub 16(%rsp), %xmm0, %xmm15
	vpminud %ymm7, %ymm1, %ymm0
	vpminud %ymm0, %ymm15, %ymm8
	vpminud 128(%r12,%r13,8), %ymm8, %ymm15
	vpminud 8(%rip), %ymm0, %ymm7
	vpminud %xmm15, %xmm9, %xmm3
	vpminud 16(%rsp), %xmm0, %xmm15
	vpminuw %ymm7, %ymm1, %ymm0
	vpminuw %ymm0, %ymm15, %ymm8
	vpminuw 128(%r12,%r13,8), %ymm8, %ymm15
	vpminuw 8(%rip), %ymm0, %ymm7
	vpminuw %xmm15, %xmm9, %xmm3
	vpminuw 16(%rsp), %xmm0, %xmm15
	vpmovmskb %ymm0, %eax
	vpmovmskb %ymm8, %r9d
	vpmovmskb %ymm15, %ecx
	vpmovmskb %ymm7, %edx
	vpmovmskb %xmm3, %r15d
	vpmovmskb %xmm15, %esi
	vpmulhw %ymm7, %ymm1, %ymm0
	vpmulhw %ymm0, %ymm15, %ymm8
	vpmulhw 128(%r12,%r13,8), %ymm8, %ymm15
	vpmulhw 8(%rip), %ymm0, %ymm7
	vpmulhw %xmm15, %xmm9, %xmm3
	vpmulhw 16(%rsp), %xmm0, %xmm15
	vpmulld %ymm7, %ymm1, %ymm0
	vpmulld %ymm0, %ymm15, %ymm8
	vpmulld 128(%r12,%r13,8), %ymm8, %ymm15
	vpmulld 8(%rip), %ymm0, %ymm7
	vpmulld %xmm15, %xmm9, %xmm3
	vpmulld 16(%rsp), %xmm0, %xmm15
	vpmullw %ymm7, %ymm1, %ymm0
	vpmullw %ymm0, %ymm15, %ymm8
	vpmullw 128(%r12,%r13,8), %ymm8, %ymm15
	vpmullw 8(%rip), %ymm0, %ymm7
	vpmullw %xmm15, %xmm9, %xmm3
	vpmullw 16(%rsp), %xmm0, %xmm15
	vpmuludq %ymm7, %ymm1, %ymm0
	vpmuludq %ymm0, %ymm15, %ymm8
	vpmuludq 128(%r12,%r13,8), %ymm8, %ymm15
	vpmuludq 8(%rip), %ymm0, %ymm7
	vpmuludq %xmm15, %xmm9, %xmm3
	vpmuludq 16(%rsp), %xmm0, %xmm15
	vpor %ymm7, %ymm1, %ymm0
	vpor %ymm0, %ymm15, %ymm8
	vpor 128(%r12,%r13,8), %ymm8, %ymm15
	vpor 8(%rip), %ymm0, %ymm7
	vpor %xmm15, %xmm9, %xmm3
	vpor 16(%rsp), %xmm0, %xmm15
	vpshufd $5, %ymm7, %ymm0
	vpshufd $42, %ymm0, %ymm8
	vpshufd $79, 128(%r12,%r13,8), %ymm15
	vpshufd $116, 8(%rip), %ymm7
	vpshufd $153, %xmm15, %xmm3
	vpshufd $190, 16(%rsp), %xmm15
	vpslld $3, %ymm1, %ymm0
	vpslld $4, %ymm15, %ymm8
	vpslld $5, %ymm8, %ymm15
	vpslld $6, %ymm0, %ymm7
	vpslld $7, %xmm9, %xmm3
	vpslld $8, %xmm0, %xmm15
	vpsllq $3, %ymm1, %ymm0
	vpsllq $4, %ymm15, %ymm8
	vpsllq $5, %ymm8, %ymm15
	vpsllq $6, %ymm0, %ymm7
	vpsllq $7, %xmm9, %xmm3
	vpsllq $8, %xmm0, %xmm15
	vpsllw $3, %ymm1, %ymm0
	vpsllw $4, %ymm15, %ymm8
	vpsllw $5, %ymm8, %ymm15
	vpsllw $6, %ymm0, %ymm7
	vpsllw $7, %xmm9, %xmm3
	vpsllw $8, %xmm0, %xmm15
	vpsrad $3, %ymm1, %ymm0
	vpsrad $4, %ymm15, %ymm8
	vpsrad $5, %ymm8, %ymm15
	vpsrad $6, %ymm0, %ymm7
	vpsrad $7, %xmm9, %xmm3
	vpsrad $8, %xmm0, %xmm15
	vpsraw $3, %ymm1, %ymm0
	vpsraw $4, %ymm15, %ymm8
	vpsraw $5, %ymm8, %ymm15
	vpsraw $6, %ymm0, %ymm7
	vpsraw $7, %xmm9, %xmm3
	vpsraw $8, %xmm0, %xmm15
	vpsrld $3, %ymm1, %ymm0
	vpsrld $4, %ymm15, %ymm8
	vpsrld $5, %ymm8, %ymm15
	vpsrld $6, %ymm0, %ymm7
	vpsrld $7, %xmm9, %xmm3
	vpsrld $8, %xmm0, %xmm15
	vpsrlq $3, %ymm1, %ymm0
	vpsrlq $4, %ymm15, %ymm8
	vpsrlq $5, %ymm8, %ymm15
	vpsrlq $6, %ymm0, %ymm7
	vpsrlq $7, %xmm9, %xmm3
	vpsrlq $8, %xmm0, %xmm15
	vpsrlw $3, %ymm1, %ymm0
	vpsrlw $4, %ymm15, %ymm8
	vpsrlw $5, %ymm8, %ymm15
	vpsrlw $6, %ymm0, %ymm7
	vpsrlw $7, %xmm9, %xmm3
	vpsrlw $8, %xmm0, %xmm15
	vpsubb %ymm7, %ymm1, %ymm0
	vpsubb %ymm0, %ymm15, %ymm8
	vpsubb 128(%r12,%r13,8), %ymm8, %ymm15
	vpsubb 8(%rip), %ymm0, %ymm7
	vpsubb %xmm15, %xmm9, %xmm3
	vpsubb 16(%rsp), %xmm0, %xmm15
	vpsubd %ymm7, %ymm1, %ymm0
	vpsubd %ymm0, %ymm15, %ymm8
	vpsubd 128(%r12,%r13,8), %ymm8, %ymm15
	vpsubd 8(%rip), %ymm0, %ymm7
	vpsubd %xmm15, %xmm9, %xmm3
	vpsubd 16(%rsp), %xmm0, %xmm15
	vpsubq %ymm7, %ymm1, %ymm0
	vpsubq %ymm0, %ymm15, %ymm8
	vpsubq 128(%r12,%r13,8), %ymm8, %ymm15
	vpsubq 8(%rip), %ymm0, %ymm7
	vpsubq %xmm15, %xmm9, %xmm3
	vpsubq 16(%rsp), %xmm0, %xmm15
	vpsubw %ymm7, %ymm1, %ymm0
	vpsubw %ymm0, %ymm15, %ymm8
	vpsubw 128(%r12,%r13,8), %ymm8, %ymm15
	vpsubw 8(%rip), %ymm0, %ymm7
	vpsubw %xmm15, %xmm9, %xmm3
	vpsubw 16(%rsp), %xmm0, %xmm15
	vptest %ymm7, %ymm0
	vptest %ymm0, %ymm8
	vptest 128(%r12,%r13,8), %ymm15
	vptest 8(%rip), %ymm7
	vptest %xmm15, %xmm3
	vptest 16(%rsp), %xmm15
	vpunpckhdq %ymm7, %ymm1, %ymm0
	vpunpckhdq %ymm0, %ymm15, %ymm8
	vpunpckhdq 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpckhdq 8(%rip), %ymm0, %ymm7
	vpunpckhdq %xmm15, %xmm9, %xmm3
	vpunpckhdq 16(%rsp), %xmm0, %xmm15
	vpunpckhqdq %ymm7, %ymm1, %ymm0
	vpunpckhqdq %ymm0, %ymm15, %ymm8
	vpunpckhqdq 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpckhqdq 8(%rip), %ymm0, %ymm7
	vpunpckhqdq %xmm15, %xmm9, %xmm3
	vpunpckhqdq 16(%rsp), %xmm0, %xmm15
	vpunpcklbw %ymm7, %ymm1, %ymm0
	vpunpcklbw %ymm0, %ymm15, %ymm8
	vpunpcklbw 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpcklbw 8(%rip), %ymm0, %ymm7
	vpunpcklbw %xmm15, %xmm9, %xmm3
	vpunpcklbw 16(%rsp), %xmm0, %xmm15
	vpunpckldq %ymm7, %ymm1, %ymm0
	vpunpckldq %ymm0, %ymm15, %ymm8
	vpunpckldq 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpckldq 8(%rip), %ymm0, %ymm7
	vpunpckldq %xmm15, %xmm9, %xmm3
	vpunpckldq 16(%rsp), %xmm0, %xmm15
	vpunpcklqdq %ymm7, %ymm1, %ymm0
	vpunpcklqdq %ymm0, %ymm15, %ymm8
	vpunpcklqdq 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpcklqdq 8(%rip), %ymm0, %ymm7
	vpunpcklqdq %xmm15, %xmm9, %xmm3
	vpunpcklqdq 16(%rsp), %xmm0, %xmm15
	vpunpcklwd %ymm7, %ymm1, %ymm0
	vpunpcklwd %ymm0, %ymm15, %ymm8
	vpunpcklwd 128(%r12,%r13,8), %ymm8, %ymm15
	vpunpcklwd 8(%rip), %ymm0, %ymm7
	vpunpcklwd %xmm15, %xmm9, %xmm3
	vpunpcklwd 16(%rsp), %xmm0, %xmm15
	vpxor %ymm7, %ymm1, %ymm0
	vpxor %ymm0, %ymm15, %ymm8
	vpxor 128(%r12,%r13,8), %ymm8, %ymm15
	vpxor 8(%rip), %ymm0, %ymm7
	vpxor %xmm15, %xmm9, %xmm3
	vpxor 16(%rsp), %xmm0, %xmm15
	vsqrtpd %ymm7, %ymm0
	vsqrtpd %ymm0, %ymm8
	vsqrtpd 128(%r12,%r13,8), %ymm15
	vsqrtpd 8(%rip), %ymm7
	vsqrtpd %xmm15, %xmm3
	vsqrtpd 16(%rsp), %xmm15
	vsqrtps %ymm7, %ymm0
	vsqrtps %ymm0, %ymm8
	vsqrtps 128(%r12,%r13,8), %ymm15
	vsqrtps 8(%rip), %ymm7
	vsqrtps %xmm15, %xmm3
	vsqrtps 16(%rsp), %xmm15
	vsubpd %ymm7, %ymm1, %ymm0
	vsubpd %ymm0, %ymm15, %ymm8
	vsubpd 128(%r12,%r13,8), %ymm8, %ymm15
	vsubpd 8(%rip), %ymm0, %ymm7
	vsubpd %xmm15, %xmm9, %xmm3
	vsubpd 16(%rsp), %xmm0, %xmm15
	vsubps %ymm7, %ymm1, %ymm0
	vsubps %ymm0, %ymm15, %ymm8
	vsubps 128(%r12,%r13,8), %ymm8, %ymm15
	vsubps 8(%rip), %ymm0, %ymm7
	vsubps %xmm15, %xmm9, %xmm3
	vsubps 16(%rsp), %xmm0, %xmm15
	vxorpd %ymm7, %ymm1, %ymm0
	vxorpd %ymm0, %ymm15, %ymm8
	vxorpd 128(%r12,%r13,8), %ymm8, %ymm15
	vxorpd 8(%rip), %ymm0, %ymm7
	vxorpd %xmm15, %xmm9, %xmm3
	vxorpd 16(%rsp), %xmm0, %xmm15
	vxorps %ymm7, %ymm1, %ymm0
	vxorps %ymm0, %ymm15, %ymm8
	vxorps 128(%r12,%r13,8), %ymm8, %ymm15
	vxorps 8(%rip), %ymm0, %ymm7
	vxorps %xmm15, %xmm9, %xmm3
	vxorps 16(%rsp), %xmm0, %xmm15
	vfmadd231ps %ymm1, %ymm9, %ymm14
	vfnmsub231ps %ymm8, %ymm2, %ymm3
	vfmadd231pd %ymm1, %ymm9, %ymm14
	vfnmsub231pd %ymm8, %ymm2, %ymm3
	vfmadd213ps %ymm1, %ymm9, %ymm14
	vfnmsub213ps %ymm8, %ymm2, %ymm3
	vfmadd213pd %ymm1, %ymm9, %ymm14
	vfnmsub213pd %ymm8, %ymm2, %ymm3
	vfmadd132ps %ymm1, %ymm9, %ymm14
	vfnmsub132ps %ymm8, %ymm2, %ymm3
	vfmadd132pd %ymm1, %ymm9, %ymm14
	# fma with a memory operand (amd64 isel folds a vector load into it)
	vfmadd231ps (%rsi), %ymm1, %ymm2
	vfmadd231ps 32(%r11), %ymm4, %ymm8
	vfmadd231pd -64(%r13,%rax,8), %ymm9, %ymm15
	vfmadd132ps (%r15), %ymm3, %ymm12
	vfmadd231ps 16(%rdi), %xmm1, %xmm2
	vfmadd132pd 0x1000(%rsp), %xmm13, %xmm0
	vfmadd231ps 8(%r12), %ymm10, %ymm11
	vfmadd231ps foo(%rip), %ymm1, %ymm2
	vfnmsub132pd %ymm8, %ymm2, %ymm3
	vzeroupper
.section .note.GNU-stack,"",@progbits
	# loops over bytes (vectorize's narrow mode)
	vpshufb .Lmask(%rip), %xmm15, %xmm3
	vpshufb (%rax), %ymm15, %ymm12
	vpshufb %xmm1, %xmm2, %xmm3
	vpshufb 32(%r9,%rcx,4), %ymm8, %ymm0
	vpmovzxbw (%rsi), %xmm4
	vpmovzxbw 8(%r13), %ymm11
	vinserti128 $1, 24(%rsi), %ymm15, %ymm15
	vpermq $8, %ymm15, %ymm15
	vmovsd %xmm15, (%rdx)
	vmovdqu %xmm15, 16(%r10)
	# AVX-512 (EVEX): the dot products, registers 16 to 31, masks, disp8*N
	vdpbf16ps %ymm3, %ymm2, %ymm1
	vdpbf16ps %xmm15, %xmm8, %xmm1
	vdpbf16ps %zmm3, %zmm2, %zmm1
	vdpbf16ps %ymm31, %ymm16, %ymm8
	vdpbf16ps %zmm15, %zmm30, %zmm17
	vdpbf16ps 64(%rax), %ymm2, %ymm1
	vdpbf16ps 33(%rax), %ymm2, %ymm1
	vdpbf16ps 4064(%rax), %ymm2, %ymm1
	vdpbf16ps -4096(%rax), %zmm2, %zmm1
	vdpbf16ps 64(%r13,%r12,4), %zmm2, %zmm1
	vdpbf16ps (%rbp), %zmm2, %zmm9
	vdpbf16ps foo(%rip), %zmm2, %zmm9
	vdpbf16ps %zmm3, %zmm2, %zmm1{%k1}
	vdpbf16ps %zmm3, %zmm2, %zmm1{%k7}{z}
	vpdpbusd %ymm3, %ymm2, %ymm1
	vpdpbusd %zmm13, %zmm12, %zmm11
	vpdpbusd 64(%r9), %ymm12, %ymm1
	{vex} vpdpbusd %ymm3, %ymm2, %ymm1
	{vex} vpdpbusd %ymm13, %ymm12, %ymm11
	{vex} vpdpbusd (%rax), %xmm2, %xmm1
	{vex} vpdpbusd 40(%r9), %ymm12, %ymm1
	# AVX-512 forms of the 512-bit emitter (zmm, k registers, EVEX-only instructions, disp8*N)
	vaddps %zmm7, %zmm8, %zmm15
	vaddps %zmm16, %zmm31, %zmm0
	vaddps 128(%rax,%rcx,4), %zmm8, %zmm16
	vaddps 64(%rsp), %zmm8, %zmm16
	vaddps -8256(%rbp), %zmm8, %zmm16
	vmulps %zmm7, %zmm8, %zmm15
	vmulps %zmm16, %zmm31, %zmm0
	vmulps 128(%rax,%rcx,4), %zmm8, %zmm16
	vmulps 64(%rsp), %zmm8, %zmm16
	vmulps -8256(%rbp), %zmm8, %zmm16
	vsubps %zmm7, %zmm8, %zmm15
	vsubps %zmm16, %zmm31, %zmm0
	vsubps 128(%rax,%rcx,4), %zmm8, %zmm16
	vsubps 64(%rsp), %zmm8, %zmm16
	vsubps -8256(%rbp), %zmm8, %zmm16
	vminps %zmm7, %zmm8, %zmm15
	vminps %zmm16, %zmm31, %zmm0
	vminps 128(%rax,%rcx,4), %zmm8, %zmm16
	vminps 64(%rsp), %zmm8, %zmm16
	vminps -8256(%rbp), %zmm8, %zmm16
	vdivps %zmm7, %zmm8, %zmm15
	vdivps %zmm16, %zmm31, %zmm0
	vdivps 128(%rax,%rcx,4), %zmm8, %zmm16
	vdivps 64(%rsp), %zmm8, %zmm16
	vdivps -8256(%rbp), %zmm8, %zmm16
	vmaxps %zmm7, %zmm8, %zmm15
	vmaxps %zmm16, %zmm31, %zmm0
	vmaxps 128(%rax,%rcx,4), %zmm8, %zmm16
	vmaxps 64(%rsp), %zmm8, %zmm16
	vmaxps -8256(%rbp), %zmm8, %zmm16
	vaddpd %zmm7, %zmm8, %zmm15
	vaddpd %zmm16, %zmm31, %zmm0
	vaddpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vaddpd 64(%rsp), %zmm8, %zmm16
	vaddpd -8256(%rbp), %zmm8, %zmm16
	vmulpd %zmm7, %zmm8, %zmm15
	vmulpd %zmm16, %zmm31, %zmm0
	vmulpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vmulpd 64(%rsp), %zmm8, %zmm16
	vmulpd -8256(%rbp), %zmm8, %zmm16
	vsubpd %zmm7, %zmm8, %zmm15
	vsubpd %zmm16, %zmm31, %zmm0
	vsubpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vsubpd 64(%rsp), %zmm8, %zmm16
	vsubpd -8256(%rbp), %zmm8, %zmm16
	vminpd %zmm7, %zmm8, %zmm15
	vminpd %zmm16, %zmm31, %zmm0
	vminpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vminpd 64(%rsp), %zmm8, %zmm16
	vminpd -8256(%rbp), %zmm8, %zmm16
	vdivpd %zmm7, %zmm8, %zmm15
	vdivpd %zmm16, %zmm31, %zmm0
	vdivpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vdivpd 64(%rsp), %zmm8, %zmm16
	vdivpd -8256(%rbp), %zmm8, %zmm16
	vmaxpd %zmm7, %zmm8, %zmm15
	vmaxpd %zmm16, %zmm31, %zmm0
	vmaxpd 128(%rax,%rcx,4), %zmm8, %zmm16
	vmaxpd 64(%rsp), %zmm8, %zmm16
	vmaxpd -8256(%rbp), %zmm8, %zmm16
	vpaddb %zmm7, %zmm8, %zmm15
	vpaddb %zmm16, %zmm31, %zmm0
	vpaddb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpaddb 64(%rsp), %zmm8, %zmm16
	vpaddb -8256(%rbp), %zmm8, %zmm16
	vpaddw %zmm7, %zmm8, %zmm15
	vpaddw %zmm16, %zmm31, %zmm0
	vpaddw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpaddw 64(%rsp), %zmm8, %zmm16
	vpaddw -8256(%rbp), %zmm8, %zmm16
	vpaddd %zmm7, %zmm8, %zmm15
	vpaddd %zmm16, %zmm31, %zmm0
	vpaddd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpaddd 64(%rsp), %zmm8, %zmm16
	vpaddd -8256(%rbp), %zmm8, %zmm16
	vpaddq %zmm7, %zmm8, %zmm15
	vpaddq %zmm16, %zmm31, %zmm0
	vpaddq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpaddq 64(%rsp), %zmm8, %zmm16
	vpaddq -8256(%rbp), %zmm8, %zmm16
	vpsubb %zmm7, %zmm8, %zmm15
	vpsubb %zmm16, %zmm31, %zmm0
	vpsubb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsubb 64(%rsp), %zmm8, %zmm16
	vpsubb -8256(%rbp), %zmm8, %zmm16
	vpsubw %zmm7, %zmm8, %zmm15
	vpsubw %zmm16, %zmm31, %zmm0
	vpsubw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsubw 64(%rsp), %zmm8, %zmm16
	vpsubw -8256(%rbp), %zmm8, %zmm16
	vpsubd %zmm7, %zmm8, %zmm15
	vpsubd %zmm16, %zmm31, %zmm0
	vpsubd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsubd 64(%rsp), %zmm8, %zmm16
	vpsubd -8256(%rbp), %zmm8, %zmm16
	vpsubq %zmm7, %zmm8, %zmm15
	vpsubq %zmm16, %zmm31, %zmm0
	vpsubq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsubq 64(%rsp), %zmm8, %zmm16
	vpsubq -8256(%rbp), %zmm8, %zmm16
	vpmullw %zmm7, %zmm8, %zmm15
	vpmullw %zmm16, %zmm31, %zmm0
	vpmullw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmullw 64(%rsp), %zmm8, %zmm16
	vpmullw -8256(%rbp), %zmm8, %zmm16
	vpmulld %zmm7, %zmm8, %zmm15
	vpmulld %zmm16, %zmm31, %zmm0
	vpmulld 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmulld 64(%rsp), %zmm8, %zmm16
	vpmulld -8256(%rbp), %zmm8, %zmm16
	vpmuludq %zmm7, %zmm8, %zmm15
	vpmuludq %zmm16, %zmm31, %zmm0
	vpmuludq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmuludq 64(%rsp), %zmm8, %zmm16
	vpmuludq -8256(%rbp), %zmm8, %zmm16
	vpminub %zmm7, %zmm8, %zmm15
	vpminub %zmm16, %zmm31, %zmm0
	vpminub 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminub 64(%rsp), %zmm8, %zmm16
	vpminub -8256(%rbp), %zmm8, %zmm16
	vpmaxub %zmm7, %zmm8, %zmm15
	vpmaxub %zmm16, %zmm31, %zmm0
	vpmaxub 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxub 64(%rsp), %zmm8, %zmm16
	vpmaxub -8256(%rbp), %zmm8, %zmm16
	vpminsw %zmm7, %zmm8, %zmm15
	vpminsw %zmm16, %zmm31, %zmm0
	vpminsw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminsw 64(%rsp), %zmm8, %zmm16
	vpminsw -8256(%rbp), %zmm8, %zmm16
	vpmaxsw %zmm7, %zmm8, %zmm15
	vpmaxsw %zmm16, %zmm31, %zmm0
	vpmaxsw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxsw 64(%rsp), %zmm8, %zmm16
	vpmaxsw -8256(%rbp), %zmm8, %zmm16
	vpminsb %zmm7, %zmm8, %zmm15
	vpminsb %zmm16, %zmm31, %zmm0
	vpminsb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminsb 64(%rsp), %zmm8, %zmm16
	vpminsb -8256(%rbp), %zmm8, %zmm16
	vpmaxsb %zmm7, %zmm8, %zmm15
	vpmaxsb %zmm16, %zmm31, %zmm0
	vpmaxsb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxsb 64(%rsp), %zmm8, %zmm16
	vpmaxsb -8256(%rbp), %zmm8, %zmm16
	vpminuw %zmm7, %zmm8, %zmm15
	vpminuw %zmm16, %zmm31, %zmm0
	vpminuw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminuw 64(%rsp), %zmm8, %zmm16
	vpminuw -8256(%rbp), %zmm8, %zmm16
	vpmaxuw %zmm7, %zmm8, %zmm15
	vpmaxuw %zmm16, %zmm31, %zmm0
	vpmaxuw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxuw 64(%rsp), %zmm8, %zmm16
	vpmaxuw -8256(%rbp), %zmm8, %zmm16
	vpminsd %zmm7, %zmm8, %zmm15
	vpminsd %zmm16, %zmm31, %zmm0
	vpminsd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminsd 64(%rsp), %zmm8, %zmm16
	vpminsd -8256(%rbp), %zmm8, %zmm16
	vpmaxsd %zmm7, %zmm8, %zmm15
	vpmaxsd %zmm16, %zmm31, %zmm0
	vpmaxsd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxsd 64(%rsp), %zmm8, %zmm16
	vpmaxsd -8256(%rbp), %zmm8, %zmm16
	vpminud %zmm7, %zmm8, %zmm15
	vpminud %zmm16, %zmm31, %zmm0
	vpminud 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminud 64(%rsp), %zmm8, %zmm16
	vpminud -8256(%rbp), %zmm8, %zmm16
	vpmaxud %zmm7, %zmm8, %zmm15
	vpmaxud %zmm16, %zmm31, %zmm0
	vpmaxud 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxud 64(%rsp), %zmm8, %zmm16
	vpmaxud -8256(%rbp), %zmm8, %zmm16
	vpunpckldq %zmm7, %zmm8, %zmm15
	vpunpckldq %zmm16, %zmm31, %zmm0
	vpunpckldq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpckldq 64(%rsp), %zmm8, %zmm16
	vpunpckldq -8256(%rbp), %zmm8, %zmm16
	vpunpckhdq %zmm7, %zmm8, %zmm15
	vpunpckhdq %zmm16, %zmm31, %zmm0
	vpunpckhdq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpckhdq 64(%rsp), %zmm8, %zmm16
	vpunpckhdq -8256(%rbp), %zmm8, %zmm16
	vpunpcklqdq %zmm7, %zmm8, %zmm15
	vpunpcklqdq %zmm16, %zmm31, %zmm0
	vpunpcklqdq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpcklqdq 64(%rsp), %zmm8, %zmm16
	vpunpcklqdq -8256(%rbp), %zmm8, %zmm16
	vpunpckhqdq %zmm7, %zmm8, %zmm15
	vpunpckhqdq %zmm16, %zmm31, %zmm0
	vpunpckhqdq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpckhqdq 64(%rsp), %zmm8, %zmm16
	vpunpckhqdq -8256(%rbp), %zmm8, %zmm16
	vpunpcklbw %zmm7, %zmm8, %zmm15
	vpunpcklbw %zmm16, %zmm31, %zmm0
	vpunpcklbw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpcklbw 64(%rsp), %zmm8, %zmm16
	vpunpcklbw -8256(%rbp), %zmm8, %zmm16
	vpunpcklwd %zmm7, %zmm8, %zmm15
	vpunpcklwd %zmm16, %zmm31, %zmm0
	vpunpcklwd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpunpcklwd 64(%rsp), %zmm8, %zmm16
	vpunpcklwd -8256(%rbp), %zmm8, %zmm16
	vpshufb %zmm7, %zmm8, %zmm15
	vpshufb %zmm16, %zmm31, %zmm0
	vpshufb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpshufb 64(%rsp), %zmm8, %zmm16
	vpshufb -8256(%rbp), %zmm8, %zmm16
	vpsllvd %zmm7, %zmm8, %zmm15
	vpsllvd %zmm16, %zmm31, %zmm0
	vpsllvd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsllvd 64(%rsp), %zmm8, %zmm16
	vpsllvd -8256(%rbp), %zmm8, %zmm16
	vpsllvq %zmm7, %zmm8, %zmm15
	vpsllvq %zmm16, %zmm31, %zmm0
	vpsllvq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsllvq 64(%rsp), %zmm8, %zmm16
	vpsllvq -8256(%rbp), %zmm8, %zmm16
	vpsrlvd %zmm7, %zmm8, %zmm15
	vpsrlvd %zmm16, %zmm31, %zmm0
	vpsrlvd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsrlvd 64(%rsp), %zmm8, %zmm16
	vpsrlvd -8256(%rbp), %zmm8, %zmm16
	vpsrlvq %zmm7, %zmm8, %zmm15
	vpsrlvq %zmm16, %zmm31, %zmm0
	vpsrlvq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsrlvq 64(%rsp), %zmm8, %zmm16
	vpsrlvq -8256(%rbp), %zmm8, %zmm16
	vpsravd %zmm7, %zmm8, %zmm15
	vpsravd %zmm16, %zmm31, %zmm0
	vpsravd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsravd 64(%rsp), %zmm8, %zmm16
	vpsravd -8256(%rbp), %zmm8, %zmm16
	vpsravq %zmm7, %zmm8, %zmm15
	vpsravq %zmm16, %zmm31, %zmm0
	vpsravq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpsravq 64(%rsp), %zmm8, %zmm16
	vpsravq -8256(%rbp), %zmm8, %zmm16
	vpandd %zmm7, %zmm8, %zmm15
	vpandd %zmm16, %zmm31, %zmm0
	vpandd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpandd 64(%rsp), %zmm8, %zmm16
	vpandd -8256(%rbp), %zmm8, %zmm16
	vpandq %zmm7, %zmm8, %zmm15
	vpandq %zmm16, %zmm31, %zmm0
	vpandq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpandq 64(%rsp), %zmm8, %zmm16
	vpandq -8256(%rbp), %zmm8, %zmm16
	vpandnd %zmm7, %zmm8, %zmm15
	vpandnd %zmm16, %zmm31, %zmm0
	vpandnd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpandnd 64(%rsp), %zmm8, %zmm16
	vpandnd -8256(%rbp), %zmm8, %zmm16
	vpandnq %zmm7, %zmm8, %zmm15
	vpandnq %zmm16, %zmm31, %zmm0
	vpandnq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpandnq 64(%rsp), %zmm8, %zmm16
	vpandnq -8256(%rbp), %zmm8, %zmm16
	vpord %zmm7, %zmm8, %zmm15
	vpord %zmm16, %zmm31, %zmm0
	vpord 128(%rax,%rcx,4), %zmm8, %zmm16
	vpord 64(%rsp), %zmm8, %zmm16
	vpord -8256(%rbp), %zmm8, %zmm16
	vporq %zmm7, %zmm8, %zmm15
	vporq %zmm16, %zmm31, %zmm0
	vporq 128(%rax,%rcx,4), %zmm8, %zmm16
	vporq 64(%rsp), %zmm8, %zmm16
	vporq -8256(%rbp), %zmm8, %zmm16
	vpxord %zmm7, %zmm8, %zmm15
	vpxord %zmm16, %zmm31, %zmm0
	vpxord 128(%rax,%rcx,4), %zmm8, %zmm16
	vpxord 64(%rsp), %zmm8, %zmm16
	vpxord -8256(%rbp), %zmm8, %zmm16
	vpxorq %zmm7, %zmm8, %zmm15
	vpxorq %zmm16, %zmm31, %zmm0
	vpxorq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpxorq 64(%rsp), %zmm8, %zmm16
	vpxorq -8256(%rbp), %zmm8, %zmm16
	vpackuswb %zmm7, %zmm8, %zmm15
	vpackuswb %zmm16, %zmm31, %zmm0
	vpackuswb 128(%rax,%rcx,4), %zmm8, %zmm16
	vpackuswb 64(%rsp), %zmm8, %zmm16
	vpackuswb -8256(%rbp), %zmm8, %zmm16
	vpackusdw %zmm7, %zmm8, %zmm15
	vpackusdw %zmm16, %zmm31, %zmm0
	vpackusdw 128(%rax,%rcx,4), %zmm8, %zmm16
	vpackusdw 64(%rsp), %zmm8, %zmm16
	vpackusdw -8256(%rbp), %zmm8, %zmm16
	vpmullq %zmm7, %zmm8, %zmm15
	vpmullq %zmm16, %zmm31, %zmm0
	vpmullq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmullq 64(%rsp), %zmm8, %zmm16
	vpmullq -8256(%rbp), %zmm8, %zmm16
	vpminsq %zmm7, %zmm8, %zmm15
	vpminsq %zmm16, %zmm31, %zmm0
	vpminsq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminsq 64(%rsp), %zmm8, %zmm16
	vpminsq -8256(%rbp), %zmm8, %zmm16
	vpmaxsq %zmm7, %zmm8, %zmm15
	vpmaxsq %zmm16, %zmm31, %zmm0
	vpmaxsq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxsq 64(%rsp), %zmm8, %zmm16
	vpmaxsq -8256(%rbp), %zmm8, %zmm16
	vpminuq %zmm7, %zmm8, %zmm15
	vpminuq %zmm16, %zmm31, %zmm0
	vpminuq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpminuq 64(%rsp), %zmm8, %zmm16
	vpminuq -8256(%rbp), %zmm8, %zmm16
	vpmaxuq %zmm7, %zmm8, %zmm15
	vpmaxuq %zmm16, %zmm31, %zmm0
	vpmaxuq 128(%rax,%rcx,4), %zmm8, %zmm16
	vpmaxuq 64(%rsp), %zmm8, %zmm16
	vpmaxuq -8256(%rbp), %zmm8, %zmm16
	vfmadd231ps %zmm7, %zmm8, %zmm15
	vfmadd231ps %zmm16, %zmm31, %zmm0
	vfmadd231ps 128(%rax,%rcx,4), %zmm8, %zmm16
	vfmadd231ps 64(%rsp), %zmm8, %zmm16
	vfmadd231ps -8256(%rbp), %zmm8, %zmm16
	vfmadd231pd %zmm7, %zmm8, %zmm15
	vfmadd231pd %zmm16, %zmm31, %zmm0
	vfmadd231pd 128(%rax,%rcx,4), %zmm8, %zmm16
	vfmadd231pd 64(%rsp), %zmm8, %zmm16
	vfmadd231pd -8256(%rbp), %zmm8, %zmm16
	vfmadd213ps %zmm7, %zmm8, %zmm15
	vfmadd213ps %zmm16, %zmm31, %zmm0
	vfmadd213ps 128(%rax,%rcx,4), %zmm8, %zmm16
	vfmadd213ps 64(%rsp), %zmm8, %zmm16
	vfmadd213ps -8256(%rbp), %zmm8, %zmm16
	vfnmadd231ps %zmm7, %zmm8, %zmm15
	vfnmadd231ps %zmm16, %zmm31, %zmm0
	vfnmadd231ps 128(%rax,%rcx,4), %zmm8, %zmm16
	vfnmadd231ps 64(%rsp), %zmm8, %zmm16
	vfnmadd231ps -8256(%rbp), %zmm8, %zmm16
	vfmsub231pd %zmm7, %zmm8, %zmm15
	vfmsub231pd %zmm16, %zmm31, %zmm0
	vfmsub231pd 128(%rax,%rcx,4), %zmm8, %zmm16
	vfmsub231pd 64(%rsp), %zmm8, %zmm16
	vfmsub231pd -8256(%rbp), %zmm8, %zmm16
	vfnmsub213pd %zmm7, %zmm8, %zmm15
	vfnmsub213pd %zmm16, %zmm31, %zmm0
	vfnmsub213pd 128(%rax,%rcx,4), %zmm8, %zmm16
	vfnmsub213pd 64(%rsp), %zmm8, %zmm16
	vfnmsub213pd -8256(%rbp), %zmm8, %zmm16
	vdpbf16ps %zmm7, %zmm8, %zmm15
	vdpbf16ps %zmm16, %zmm31, %zmm0
	vdpbf16ps 128(%rax,%rcx,4), %zmm8, %zmm16
	vdpbf16ps 64(%rsp), %zmm8, %zmm16
	vdpbf16ps -8256(%rbp), %zmm8, %zmm16
	vpdpbusd %zmm7, %zmm8, %zmm15
	vpdpbusd %zmm16, %zmm31, %zmm0
	vpdpbusd 128(%rax,%rcx,4), %zmm8, %zmm16
	vpdpbusd 64(%rsp), %zmm8, %zmm16
	vpdpbusd -8256(%rbp), %zmm8, %zmm16
	vpcmpeqb %zmm1, %zmm2, %k1
	vpcmpeqb %zmm17, %zmm30, %k7
	vpcmpeqb 64(%rax), %zmm9, %k2
	vpcmpeqw %zmm1, %zmm2, %k1
	vpcmpeqw %zmm17, %zmm30, %k7
	vpcmpeqw 64(%rax), %zmm9, %k2
	vpcmpeqd %zmm1, %zmm2, %k1
	vpcmpeqd %zmm17, %zmm30, %k7
	vpcmpeqd 64(%rax), %zmm9, %k2
	vpcmpeqq %zmm1, %zmm2, %k1
	vpcmpeqq %zmm17, %zmm30, %k7
	vpcmpeqq 64(%rax), %zmm9, %k2
	vpcmpgtb %zmm1, %zmm2, %k1
	vpcmpgtb %zmm17, %zmm30, %k7
	vpcmpgtb 64(%rax), %zmm9, %k2
	vpcmpgtw %zmm1, %zmm2, %k1
	vpcmpgtw %zmm17, %zmm30, %k7
	vpcmpgtw 64(%rax), %zmm9, %k2
	vpcmpgtd %zmm1, %zmm2, %k1
	vpcmpgtd %zmm17, %zmm30, %k7
	vpcmpgtd 64(%rax), %zmm9, %k2
	vpcmpgtq %zmm1, %zmm2, %k1
	vpcmpgtq %zmm17, %zmm30, %k7
	vpcmpgtq 64(%rax), %zmm9, %k2
	vcmpps $0, %zmm1, %zmm2, %k1
	vcmpps $1, %zmm1, %zmm2, %k1
	vcmpps $2, %zmm1, %zmm2, %k1
	vcmpps $3, %zmm1, %zmm2, %k1
	vcmpps $1, %zmm17, %zmm12, %k3
	vcmppd $0, %zmm1, %zmm2, %k1
	vcmppd $1, %zmm1, %zmm2, %k1
	vcmppd $2, %zmm1, %zmm2, %k1
	vcmppd $3, %zmm1, %zmm2, %k1
	vcmppd $1, %zmm17, %zmm12, %k3
	vpmovm2b %k1, %zmm3
	vpmovm2b %k7, %zmm19
	vpmovm2w %k1, %zmm3
	vpmovm2w %k7, %zmm19
	vpmovm2d %k1, %zmm3
	vpmovm2d %k7, %zmm19
	vpmovm2q %k1, %zmm3
	vpmovm2q %k7, %zmm19
	vpmovb2m %zmm3, %k1
	vpmovw2m %zmm3, %k1
	vpmovd2m %zmm3, %k1
	vpmovq2m %zmm3, %k1
	vpternlogd $255, %zmm1, %zmm2, %zmm3
	vpternlogq $255, %zmm21, %zmm2, %zmm13
	vpternlogd $202, %zmm1, %zmm2, %zmm3
	vpternlogq $202, %zmm21, %zmm2, %zmm13
	vpternlogd $150, %zmm1, %zmm2, %zmm3
	vpternlogq $150, %zmm21, %zmm2, %zmm13
	vsqrtps %zmm1, %zmm2
	vsqrtps 64(%rdi), %zmm22
	vsqrtpd %zmm1, %zmm2
	vsqrtpd 64(%rdi), %zmm22
	vcvtdq2ps %zmm1, %zmm2
	vcvtdq2ps 64(%rdi), %zmm22
	vcvttps2dq %zmm1, %zmm2
	vcvttps2dq 64(%rdi), %zmm22
	vcvtudq2ps %zmm1, %zmm2
	vcvtudq2ps 64(%rdi), %zmm22
	vcvttps2udq %zmm1, %zmm2
	vcvttps2udq 64(%rdi), %zmm22
	vcvtqq2pd %zmm1, %zmm2
	vcvtqq2pd 64(%rdi), %zmm22
	vcvtuqq2pd %zmm1, %zmm2
	vcvtuqq2pd 64(%rdi), %zmm22
	vcvttpd2qq %zmm1, %zmm2
	vcvttpd2qq 64(%rdi), %zmm22
	vcvttpd2uqq %zmm1, %zmm2
	vcvttpd2uqq 64(%rdi), %zmm22
	vrndscalepd $3, %zmm1, %zmm2
	vrndscalepd $1, 128(%rdi), %zmm2
	vrndscaleps $3, %zmm1, %zmm2
	vrndscaleps $1, 128(%rdi), %zmm2
	vpshufd $3, %zmm1, %zmm2
	vpshufd $1, 128(%rdi), %zmm2
	vpsllw $5, %zmm1, %zmm2
	vpsllw $31, %zmm17, %zmm9
	vpslld $5, %zmm1, %zmm2
	vpslld $31, %zmm17, %zmm9
	vpsllq $5, %zmm1, %zmm2
	vpsllq $31, %zmm17, %zmm9
	vpsrlw $5, %zmm1, %zmm2
	vpsrlw $31, %zmm17, %zmm9
	vpsrld $5, %zmm1, %zmm2
	vpsrld $31, %zmm17, %zmm9
	vpsrlq $5, %zmm1, %zmm2
	vpsrlq $31, %zmm17, %zmm9
	vpsraw $5, %zmm1, %zmm2
	vpsraw $31, %zmm17, %zmm9
	vpsrad $5, %zmm1, %zmm2
	vpsrad $31, %zmm17, %zmm9
	vpsraq $5, %zmm1, %zmm2
	vpsraq $31, %zmm17, %zmm9
	vpbroadcastb %xmm15, %zmm3
	vpbroadcastb 3(%rax), %zmm3
	vpbroadcastb 512(%rax,%rdx,4), %zmm19
	vpbroadcastb (%rsi), %ymm3
	vpbroadcastw %xmm15, %zmm3
	vpbroadcastw 6(%rax), %zmm3
	vpbroadcastw 512(%rax,%rdx,4), %zmm19
	vpbroadcastw (%rsi), %ymm3
	vpbroadcastd %xmm15, %zmm3
	vpbroadcastd 12(%rax), %zmm3
	vpbroadcastd 512(%rax,%rdx,4), %zmm19
	vpbroadcastd (%rsi), %ymm3
	vpbroadcastq %xmm15, %zmm3
	vpbroadcastq 24(%rax), %zmm3
	vpbroadcastq 512(%rax,%rdx,4), %zmm19
	vpbroadcastq (%rsi), %ymm3
	vbroadcastss %xmm15, %zmm3
	vbroadcastss 12(%rax), %zmm3
	vbroadcastss 512(%rax,%rdx,4), %zmm19
	vbroadcastss (%rsi), %ymm3
	vbroadcastsd %xmm15, %zmm3
	vbroadcastsd 24(%rax), %zmm3
	vbroadcastsd 512(%rax,%rdx,4), %zmm19
	vbroadcastsd (%rsi), %ymm3
	vbroadcasti32x4 32(%rax), %zmm3
	vbroadcasti32x4 64(%rax), %zmm23
	vbroadcasti64x4 32(%rax), %zmm3
	vbroadcasti64x4 64(%rax), %zmm23
	vpmovsxbw %ymm1, %zmm2
	vpmovsxbw 64(%rdi), %zmm2
	vpmovsxbw 32(%rdi), %zmm2
	vpmovsxwd %ymm1, %zmm2
	vpmovsxwd 64(%rdi), %zmm2
	vpmovsxwd 32(%rdi), %zmm2
	vpmovsxdq %ymm1, %zmm2
	vpmovsxdq 64(%rdi), %zmm2
	vpmovsxdq 32(%rdi), %zmm2
	vpmovzxbw %ymm1, %zmm2
	vpmovzxbw 64(%rdi), %zmm2
	vpmovzxbw 32(%rdi), %zmm2
	vpmovzxwd %ymm1, %zmm2
	vpmovzxwd 64(%rdi), %zmm2
	vpmovzxwd 32(%rdi), %zmm2
	vpmovzxdq %ymm1, %zmm2
	vpmovzxdq 64(%rdi), %zmm2
	vpmovzxdq 32(%rdi), %zmm2
	vcvtps2pd %ymm1, %zmm2
	vcvtps2pd 64(%rdi), %zmm2
	vcvtps2pd 32(%rdi), %zmm2
	vcvtph2ps %ymm1, %zmm2
	vcvtph2ps 64(%rdi), %zmm2
	vcvtph2ps 32(%rdi), %zmm2
	vcvtpd2ps %zmm1, %ymm2
	vcvtpd2ps 128(%rax), %ymm2
	vcvtps2ph $4, %zmm1, %ymm2
	vcvtps2ph $4, %zmm1, 32(%rax)
	vpmovwb %zmm1, %ymm2
	vpmovwb %zmm17, 32(%rax)
	vpmovdw %zmm1, %ymm2
	vpmovdw %zmm17, 32(%rax)
	vpmovqd %zmm1, %ymm2
	vpmovqd %zmm17, 32(%rax)
	vextracti32x4 $0, %zmm3, %xmm15
	vextracti32x4 $1, %zmm3, %xmm15
	vextracti32x4 $2, %zmm3, %xmm15
	vextracti32x4 $3, %zmm3, %xmm15
	vextracti32x4 $2, %zmm3, 48(%rax)
	vextractf32x4 $0, %zmm3, %xmm15
	vextractf32x4 $1, %zmm3, %xmm15
	vextractf32x4 $2, %zmm3, %xmm15
	vextractf32x4 $3, %zmm3, %xmm15
	vextractf32x4 $2, %zmm3, 48(%rax)
	vextracti64x4 $1, %zmm3, %ymm15
	vextracti64x4 $1, %zmm3, 64(%rax)
	vextractf64x4 $1, %zmm3, %ymm15
	vextractf64x4 $1, %zmm3, 64(%rax)
	vinserti32x4 $3, %xmm15, %zmm2, %zmm3
	vinserti32x4 $1, 48(%rax), %zmm2, %zmm3
	vinsertf32x4 $3, %xmm15, %zmm2, %zmm3
	vinsertf32x4 $1, 48(%rax), %zmm2, %zmm3
	vinserti64x4 $1, %ymm15, %zmm2, %zmm3
	vinserti64x4 $1, 64(%rax), %zmm2, %zmm3
	vinsertf64x4 $1, %ymm15, %zmm2, %zmm3
	vinsertf64x4 $1, 64(%rax), %zmm2, %zmm3
	vmovups %zmm1, %zmm2
	vmovups %zmm17, %zmm2
	vmovups 64(%rax), %zmm3
	vmovups %zmm3, -128(%rbp)
	vmovups 100(%rsp), %zmm3
	vmovups %zmm3, 4096(%rsp)
	vmovups %zmm3, .Lx(%rip)
	vmovaps %zmm1, %zmm2
	vmovaps %zmm17, %zmm2
	vmovaps 64(%rax), %zmm3
	vmovaps %zmm3, -128(%rbp)
	vmovaps 100(%rsp), %zmm3
	vmovaps %zmm3, 4096(%rsp)
	vmovaps %zmm3, .Lx(%rip)
	vmovupd %zmm1, %zmm2
	vmovupd %zmm17, %zmm2
	vmovupd 64(%rax), %zmm3
	vmovupd %zmm3, -128(%rbp)
	vmovupd 100(%rsp), %zmm3
	vmovupd %zmm3, 4096(%rsp)
	vmovupd %zmm3, .Lx(%rip)
	vmovapd %zmm1, %zmm2
	vmovapd %zmm17, %zmm2
	vmovapd 64(%rax), %zmm3
	vmovapd %zmm3, -128(%rbp)
	vmovapd 100(%rsp), %zmm3
	vmovapd %zmm3, 4096(%rsp)
	vmovapd %zmm3, .Lx(%rip)
	vpaddd %zmm1, %zmm2, %zmm3{%k1}{z}
	vmovups %zmm1, %zmm2{%k3}
	vpxor %xmm3, %xmm3, %xmm3
	vpandq %ymm3, %ymm4, %ymm5
	vpaddd %ymm17, %ymm4, %ymm5
	vaddps %xmm17, %xmm4, %xmm5
.Lx:
.Lmask:
	.byte 0
