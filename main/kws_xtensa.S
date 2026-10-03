/*
 * kws_xtensa.S -- the INT8 reduction for the ESP32-S3.
 *
 * WHY ASSEMBLY AND NOT INLINE ASM
 * --------------------------------
 * EE.VMULAS.S8.ACCX takes and returns Q registers. GCC's Xtensa back end does
 * not model them, so it cannot allocate one for an inline-asm operand: with an
 * "r" constraint it puts a general register there and the assembler rejects it
 * ("bad register name"), and with an explicit register variable it rejects the
 * name outright ("invalid register name for 'acc'"). A .S file is the only way
 * to name a Q register, which is why this is a .S file.
 *
 * WHAT IT BUYS
 * ------------
 * One EE.VMULAS.S8.ACCX is sixteen signed 8x8-bit multiply-accumulates: eight
 * in the low 64 bits of the accumulator and eight in the high, fed by two 128-bit
 * loads. Plain C cannot get anywhere near it. There is no sign-extending byte
 * load on this core, so a scalar int8 multiply-accumulate costs a load, a
 * two-instruction sign extension, a multiply and an add -- four instructions
 * per MAC, worse than float32, which has hardware FMUL.S and FADD.S.
 *
 * This is the difference between the conv2 + conformer GEMMs being 85% of the
 * network's cost and being a rounding error in it.
 *
 * THE ALIGNMENT CONTRACT
 * ----------------------
 * Both operands must be 16-byte aligned and the length a multiple of 16. That
 * is arranged offline, not here:
 *
 *   - tools/export_weights.py pads every weight row to a multiple of 16 with
 *     zeros and starts every tensor on a 16-byte boundary, so consecutive
 *     output rows are all aligned;
 *   - kws_model.c pads every quantised activation row to a multiple of 16.
 *
 * A dot product of n/16 vector MACs then needs no shifting, no tail and no
 * unaligned-load handling. The zero pad bytes contribute nothing to the sum.
 *
 * kws_int8.c re-checks both preconditions before calling in, and falls back to
 * the portable loop if either fails, so this file is an optimisation and never
 * a correctness requirement.
 *
 * Reference for the instruction semantics and the two-cycle gap before reading
 * the accumulator: Espressif's esp-nn component, src/common/esp_nn_dot_s8_esp32s3.S.
 */

    .text
    .align  4

/*
 * int32_t kws_asm_dot_i8(const int8_t *a, const int8_t *b, int n)
 *
 *   in   a2 = a, 16-byte aligned
 *        a3 = b, 16-byte aligned
 *        a4 = n, a non-zero multiple of 16
 *   out  a2 = sum over i of a[i] * b[i]
 *
 * The windowed ABI: the caller's register window is saved with the 32-byte
 * frame this function's callees would need, and `retw.n` restores it.
 */
    .type   kws_asm_dot_i8, @function
    .align  4
    .global kws_asm_dot_i8

kws_asm_dot_i8:
    entry   a1, 32

    ee.zero.accx                /* accumulator is 64-bit; low 32 is the answer */
    beqz    a4, .Ldone

    srli    a5, a4, 4           /* a5 = number of 16-byte blocks, minus 1 */
    addi    a5, a5, -1

    /* Prime: q0 = a[0..15], q1 = b[0..15]. */
    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a3, 16

    beqz    a5, .Llast

    /*
     * Steady state: the fused form multiplies the pair already in the
     * registers and loads the NEXT `a` block in the same instruction, so the
     * only extra work per block is loading `b`. `b` runs one block ahead,
     * which is why the final block is handled outside the loop.
     */
    loopgtz a5, .Lloop_end
.Lloop:
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    ee.vld.128.ip             q1, a3, 16
.Lloop_end:

.Llast:
    ee.vmulas.s8.accx  q0, q1

.Ldone:
    /* Reading ACCX costs the pipeline two cycles. */
    nop
    nop
    rur.accx_0 a2

    retw.n

    .size   kws_asm_dot_i8, . - kws_asm_dot_i8

/*
 * void kws_asm_dot_taps_i8(const int8_t *a, const int8_t *b,
 *                          int n, int ntaps, int32_t *out)
 *
 *   in   a2 = a, 16-byte aligned, ntaps * n bytes
 *        a3 = b, 16-byte aligned, ntaps * n bytes
 *        a4 = n, bytes PER TAP, a non-zero multiple of 16
 *        a5 = ntaps, taps in this window
 *        a6 = out, ntaps int32 to write
 *   out  out[k] = sum over i of a[k*n+i] * b[k*n+i]
 *
 * One int32 per tap, one call for the whole window. The caller applies the
 * per-tap activation scales.
 *
 * WHY THIS EXISTS
 * --------------
 * conv2's window is nine taps of 32 channels and each tap carries its own
 * activation scale. Reducing the taps one call at a time -- nine calls per
 * output channel -- costs 211,680 calls per inference for 423,360 useful
 * vector MACs. Each call pays a ~15-instruction prologue (window save,
 * zero.accx, block count, prime, the two-cycle accumulator read, retw.n) for
 * two useful EE.VMULAS: a 7:1 overhead-to-work ratio, which measured as conv2
 * taking 71.8 ms against a 1.8 ms arithmetic ideal.
 *
 * The taps are walked inside one call, so the prologue is paid once per window
 * instead of once per tap -- 23,520 calls rather than 211,680.
 *
 * NO FLOAT INSTRUCTIONS
 * ---------------------
 * The per-tap scaling is nine multiplies and adds for the caller, which is
 * nothing next to the 288 byte reductions it is scaling, and keeping this file
 * free of FPU opcodes means it assembles with the same instruction subset
 * kws_asm_dot_i8 already uses. There are no t0-t3 registers outside ESP-IDF's
 * macro headers, so the working registers are the caller-saved a2..a7 and
 * nothing is spilled to the stack.
 *
 * The accumulator is cleared per tap because each tap is a separate result.
 * a2 and a3 walk forward through the window by n bytes and a5 counts down.
 */
    .type   kws_asm_dot_taps_i8, @function
    .align  4
    .global kws_asm_dot_taps_i8

kws_asm_dot_taps_i8:
    entry   a1, 32

.Ltap_loop:
    ee.zero.accx

    /* Register budget: a2/a3 walk the operands, a4 is the per-tap length, a5
     * counts taps, a6 walks the output. That leaves a7, which does double duty
     * as the block counter inside the inner loop and as the accumulator
     * landing register afterwards -- safe because the two uses do not overlap.
     * a5 cannot be the block counter precisely because it has to survive the
     * whole window, and `loopgtz` needs a register the caller cannot see. */
    srli    a7, a4, 4          /* 16-byte blocks per tap */
    addi    a7, a7, -1

    ee.vld.128.ip   q0, a2, 16
    ee.vld.128.ip   q1, a3, 16
    beqz    a7, .Ltap_last

.Ltap_blocks:
    ee.vmulas.s8.accx.ld.ip  q0, a2, 16, q0, q1
    ee.vld.128.ip             q1, a3, 16
    addi    a7, a7, -1
    bnez    a7, .Ltap_blocks

.Ltap_last:
    ee.vmulas.s8.accx  q0, q1

    nop                       /* reading ACCX costs the pipeline two cycles */
    nop
    rur.accx_0 a7             /* low 32 bits are this tap's sum */
    s32i    a7, a6, 0
    addi    a6, a6, 4

    /* NO POINTER ADVANCE HERE.
     *
     * The vector loads are post-incrementing: the prime loads 16 bytes and the
     * block loop loads 16 more per iteration, so by the time the tap's
     * reduction is finished a2 and a3 already point at the START of the next
     * tap -- exactly n bytes on, because a tap is n/16 blocks of 16. Adding a
     * stride here as well skips every other tap and reads into the following
     * output channel's weights. That bug cost a 6.8x slowdown, because the
     * self test caught it, which demoted the whole kernel to the portable
     * scalar loop. */
    addi    a5, a5, -1
    bnez    a5, .Ltap_loop

    retw.n

    .size   kws_asm_dot_taps_i8, . - kws_asm_dot_taps_i8