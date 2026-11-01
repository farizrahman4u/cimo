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
.section .note.GNU-stack,"",@progbits
