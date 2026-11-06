/* Every instruction form, operand form and directive cimo.backend's arm64 emitter can produce
   (python/cimo/qbe/arm64/emit.py, python/cimo/qbe/emit.py), in ELF syntax.
   test_assembler.py assembles this with GNU as and with cimo.assembler and compares the objects. */
.text
.balign 16
.globl forms
forms:
	hint	#34
	/* prologues (arm64_emitfn), all four frame-size cases, and the vararg save area */
	str	q7, [sp, -16]!
	str	q0, [sp, -16]!
	stp	x6, x7, [sp, -16]!
	stp	x0, x1, [sp, -16]!
	stp	x29, x30, [sp, -512]!
	stp	x29, x30, [sp, -16]!
	sub	sp, sp, #4095
	mov	x16, #65535
	sub	sp, sp, x16
	mov	x16, #4464
	movk	x16, #1, lsl #16
	mov	x29, sp
	/* epilogues */
	ldp	x29, x30, [sp], 504
	ldp	x29, x30, [sp], 16
	add	sp, sp, #4095
	mov	x16, #65535
	add	sp, sp, x16
	mov	sp, x29
	ret
	/* omap: integer ops, w and x, register and immediate second operands */
	add	w0, w1, w2
	add	x0, x1, x2
	add	x28, x27, x30
	add	w3, w4, #0
	add	x3, x4, #4095
	add	x5, x6, #1, lsl #12
	add	w5, w6, #4095, lsl #12
	add	x0, x29, #4095
	add	x9, sp, #16
	add	x9, sp, x10
	sub	w0, w1, w2
	sub	x0, x1, x2
	sub	x3, x4, #17
	sub	w3, w4, #2, lsl #12
	sub	sp, sp, x9
	sub	sp, sp, #64
	neg	w0, w1
	neg	x0, x1
	and	w0, w1, w2
	and	x0, x1, x2
	and	w0, w1, #255
	and	x0, x1, #0xff00ff00ff00ff00
	and	x0, x1, #16777216
	and	w0, w1, #2147483648
	orr	w0, w1, w2
	orr	x0, x1, x2
	orr	x7, x8, #-16777216
	eor	w0, w1, w2
	eor	x0, x1, x2
	eor	x0, x1, #6148914691236517205
	asr	w0, w1, w2
	asr	x0, x1, x2
	asr	w0, w1, #31
	asr	x0, x1, #63
	asr	x0, x1, #0
	lsr	w0, w1, w2
	lsr	x0, x1, x2
	lsr	w0, w1, #1
	lsr	x0, x1, #32
	lsl	w0, w1, w2
	lsl	x0, x1, x2
	lsl	w0, w1, #31
	lsl	x0, x1, #1
	lsl	x0, x1, #0
	mul	w0, w1, w2
	mul	x0, x1, x2
	sdiv	w0, w1, w2
	sdiv	x0, x1, x2
	udiv	w0, w1, w2
	udiv	x0, x1, x2
	sdiv	w17, w0, w1
	msub	w0, w17, w1, w0
	udiv	x17, x0, x1
	msub	x0, x17, x1, x0
	mov	w0, w1
	mov	x0, x1
	mov	x17, x0
	mov	x0, x1
	mov	x1, x17
	mov	x0, sp
	cmp	w0, w1
	cmp	x0, x1
	cmp	w0, #0
	cmp	x0, #4095
	cmp	x0, #1, lsl #12
	cmn	w0, w1
	cmn	x0, #7
	cmn	x0, #5, lsl #12
	/* loadcon: mov wide / bitmask immediates and movk sequences */
	mov	w0, #0
	mov	x0, #-1
	mov	w0, #-1
	mov	x0, #-2
	mov	w0, #-65536
	mov	x0, #-65536
	mov	w0, #65535
	mov	x0, #65535
	mov	x0, #-4294967296
	mov	w0, #252645135
	mov	x0, #1085102592571150095
	mov	x0, #4096
	movk	x0, #0x1, lsl #16
	movk	x0, #0xffff, lsl #32
	movk	x0, #0x8000, lsl #48
	mov	w1, #4660
	movk	w1, #0x5678, lsl #16
	mov	w0, #-30875
	mov	x0, #-30875
	/* Oaddr */
	add	x3, x29, #4095
	mov	x3, #65535
	add	x3, x29, x3
	mov	x3, #1
	movk	x3, #1, lsl #16
	add	x3, x29, x3
	/* extensions */
	sxtb	w0, w1
	sxtb	x0, w1
	uxtb	w0, w1
	sxth	w0, w1
	sxth	x0, w1
	uxth	w0, w1
	sxtw	x0, w1
	mov	w0, w1
	/* floating point */
	fadd	s0, s1, s2
	fadd	d0, d1, d2
	fsub	s0, s1, s2
	fsub	d0, d1, d2
	fmul	s0, s1, s2
	fmul	d0, d1, d2
	fdiv	s0, s1, s2
	fdiv	d0, d1, d2
	fneg	s0, s1
	fneg	d30, d31
	fmov	s0, s1
	fmov	d0, d1
	fmov	s31, s0
	fmov	d0, d31
	fmov	w0, s1
	fmov	x0, d1
	fmov	s0, w1
	fmov	d0, x1
	fmov	d31, x17
	fmov	s31, w17
	fcvt	d0, s1
	fcvt	s0, d1
	fcvtzs	w0, s1
	fcvtzs	x0, s1
	fcvtzs	w0, d1
	fcvtzs	x0, d1
	fcvtzu	w0, s1
	fcvtzu	x0, s1
	fcvtzu	w0, d1
	fcvtzu	x0, d1
	scvtf	s0, w1
	scvtf	d0, w1
	scvtf	s0, x1
	scvtf	d0, x1
	ucvtf	s0, w1
	ucvtf	d0, w1
	ucvtf	s0, x1
	ucvtf	d0, x1
	fcmpe	s0, s1
	fcmpe	d0, d1
	fcmpe	d31, d0
	/* flags */
	cset	w0, eq
	cset	x0, ne
	cset	w0, ge
	cset	w0, gt
	cset	w0, le
	cset	w0, lt
	cset	w0, cs
	cset	w0, hi
	cset	w0, ls
	cset	w0, cc
	cset	w0, mi
	cset	w0, vc
	cset	w0, vs
	/* loads and stores: register, scaled, unscaled (ldur/stur) and slot offsets */
	strb	w0, [x1]
	strb	w0, [x1, 4095]
	strb	w0, [x1, -1]
	strh	w0, [x1]
	strh	w0, [x1, 8190]
	strh	w0, [x1, 3]
	str	w0, [x1]
	str	w0, [x1, 16380]
	str	w0, [x1, 2]
	str	w0, [x1, -256]
	str	x0, [x1]
	str	x0, [x1, 32760]
	str	x0, [x1, 255]
	str	x0, [x29, 16]
	str	x0, [sp]
	str	x0, [sp, 8]
	str	s0, [x1]
	str	s0, [x1, 16380]
	str	s31, [x1, 4]
	str	d0, [x1]
	str	d0, [x29, 32760]
	str	d0, [x1, -8]
	ldrsb	w0, [x1]
	ldrsb	x0, [x1, 7]
	ldrb	w0, [x1]
	ldrb	w0, [x1, 4095]
	ldrsh	w0, [x1, 2]
	ldrsh	x0, [x1, -2]
	ldrh	w0, [x1, 8190]
	ldr	w0, [x1]
	ldr	w0, [x1, 4]
	ldrsw	x0, [x1]
	ldrsw	x0, [x1, 16380]
	ldr	x0, [x1]
	ldr	x0, [x1, 8]
	ldr	x0, [x1, -8]
	ldr	x0, [x29, 24]
	ldr	s0, [x1, 4]
	ldr	d0, [x1, 8]
	ldr	d8, [x29, 40]
	str	d8, [x29, 40]
	str	x19, [x29, 48]
	/* calls and branches */
	blr	x0
	blr	x16
	bl	ext
	bl	forms
	bl	local_fn
	brk	#1000
.L1:
	beq	.L1
	bne	.L2
	bge	.L1
	bgt	.L2
	ble	.L1
	blt	.L2
	bcs	.L1
	bhi	.L2
	bls	.L1
	bcc	.L2
	bmi	.L1
	bpl	.L2
	bvc	.L1
	bvs	.L2
	b	.L1
	b	.L2
.L2:
	/* addresses: globals, locals, other sections, offsets, GOT */
	adrp	x0, glob
	add	x0, x0, #:lo12:glob
	adrp	x1, glob+8
	add	x1, x1, #:lo12:glob+8
	adrp	x2, glob+-8
	add	x2, x2, #:lo12:glob+-8
	adrp	x3, ".Lfp0"
	add	x3, x3, #:lo12:".Lfp0"
	adrp	x4, loc
	add	x4, x4, #:lo12:loc
	adrp	x5, cm
	add	x5, x5, #:lo12:cm
	adrp	x6, :got:ext
	ldr	x6, [x6, #:got_lo12:ext]
	adrp	x7, :got:extdata
	ldr	x7, [x7, #:got_lo12:extdata]
	adrp	x8, bssvar
	add	x8, x8, #:lo12:bssvar
	adrp	x9, local_fn
	add	x9, x9, #:lo12:local_fn
	ret
.type forms, @function
.size forms, .-forms
/* end function forms */

.text
local_fn:
	hint	#34
	stp	x29, x30, [sp, -16]!
	mov	x29, sp
	ldp	x29, x30, [sp], 16
	ret
.balign 16
.globl after_align
after_align:
	ret

.data
.balign 8
.globl glob
glob:
	.quad 0
	.quad -1
	.quad glob+0
	.quad loc+8
	.quad ext+0
	.quad ext-16
	.int 4294967295
	.int glob+4
	.short 65535
	.short -1
	.byte 255
	.byte 0
	.ascii "a\"b\\c\n\t\001\177\377\x41z"
	.fill 3,1,0
loc:
	.byte 1
	.balign 4
	.int 7
.data
.balign 16
	.ascii ""
	.quad 9
.bss
.balign 8
bssvar:
	.fill 24,1,0
.globl bssglob
bssglob:
	.fill 8,1,0
.comm cm,16,8
.comm cm2,4
.section .rodata
.p2align 3
.Lfp0:
	.quad 4607182418800017408 /* 1.000000 */

.section .rodata
.p2align 2
.Lfp1:
	.int 1065353216 /* 1.000000 */
.section .data.custom,"aw"
	.byte 3
.section .text.custom,"ax"
.globl t2
t2:
	ret
.section .note.GNU-stack,"",@progbits
	// loops over bytes (vectorize's narrow mode)
	ld3	{v29.8b, v30.8b, v31.8b}, [x1]
	ld3	{v0.8b, v1.8b, v2.8b}, [sp]
	ld3	{v30.8b, v31.8b, v0.8b}, [x28]
	ushll	v3.8h, v29.8b, #0
	ushll	v31.8h, v0.8b, #0
	xtn	v31.8b, v7.8h
	xtn	v0.8b, v31.8h
	ldr	d31, [x2]
	str	d31, [x3, 8]
	ld3	{v28.8b, v29.8b, v30.8b}, [x5]
	umull	v3.8h, v29.8b, v0.8b
	umlal	v3.8h, v28.8b, v1.8b
	umlal	v31.8h, v30.8b, v17.8b
	shrn	v31.8b, v3.8h, #8
	shrn	v0.8b, v12.8h, #1
	ld3	{v28.16b, v29.16b, v30.16b}, [x1]
	umull2	v3.8h, v29.16b, v0.16b
	umlal2	v3.8h, v28.16b, v1.16b
	ushll2	v5.8h, v30.16b, #0
