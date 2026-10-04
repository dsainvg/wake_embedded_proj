/*
 * m1_g_wide inference for the "amaze" keyword.
 *
 * Port of models/models_torch_hc.py :: TimedKWSNet with the
 * ARCHS_M1["m1_g_wide"] preset: dim 96, stem 32, 2 blocks, 4 heads,
 * ffn_mult 1.0, time_stride 4, mel_pool_rank 8. Input is the (49, 40)
 * log-mel from kws_frontend.h; output is P(keyword) in [0, 1].
 *
 * WHY THIS REUSES kws_int8.h INSTEAD OF ESP-DL
 * --------------------------------------------
 * ESP-DL v3.3.13 targets the ESP32-S3 and has AI-instruction kernels for
 * exactly these ops (Conv2d, DepthwiseConv2D, Gemm), plus a static memory
 * planner and dual-core Conv2D. It was evaluated and rejected, for reasons
 * that are not "it is slower":
 *
 *   1. Its only supported entry point is a FlatBuffers `.espdl` graph, produced
 *      by esp-ppq, which needs PyTorch. No torch on the build host, so the model
 *      file cannot be generated at all.
 *   2. It exposes no standalone C kernel API. The ops are `dl::` classes bound
 *      to that graph and to `dl::TensorBase`, so using a kernel means adopting
 *      the framework, its loader and its own memory planner -- a ~10 MB
 *      component whose planner allocates per whole model, which is the wrong
 *      tool for a hard internal-RAM ceiling this code has to stay under.
 *   3. It has no GroupNorm, and this architecture has GroupNorm in all three
 *      stem blocks. Those would be custom ops regardless.
 *
 * ESP-NN, the older standalone C library with `esp_nn_*_s8` entry points, would
 * have fit the calling convention exactly, but IDF 6.1 no longer bundles it and
 * ESP-DL 3.3.13 carries no `esp_nn_*` symbols.
 *
 * What IS reused is better than either: main/kws_int8.h already contains an
 * Xtensa AI-instruction int8 reduction (128-bit loads, kws_xtensa.S) with a
 * portable twin and a boot-time self-test that demotes the assembly if the two
 * disagree. That kernel has been verified bit-comparable to JAX across six
 * fixtures. ~95% of this model's MACs are plain int8 GEMM, which is what
 * kws_accum_row_i8() already does. Only the architecture-specific glue is new.
 *
 * ARITHMETIC
 * ----------
 * 2,771,692 MAC, walked from the declared layer sequence by
 * tools/export_m1gw_weights.py. At 16 MAC/instruction and 240 MHz that is a
 * 0.722 ms arithmetic floor, against 11.11 ms of budget at a 100 ms hop.
 *
 * The checkpoint records 1,473,962 MAC for this architecture. That figure is
 * 1.88x low; see docs/latency.md and the exporter's own report. The walk is
 * used because it comes from the shapes this code actually executes.
 *
 * RAM
 * ---
 * Activations are int8 end to end. Every GEMM output is requantised to int8
 * immediately, row by row, via kws_accum_row_i8(), so no full float32
 * activation tensor is ever materialised -- the largest single float buffer is
 * the GroupNorm scratch. See KWS_M1GW_RAM_BYTES for the audited total, which is
 * asserted at compile time against the 256 KB ceiling.
 */

#ifndef KWS_M1GW_H
#define KWS_M1GW_H

#include <stddef.h>
#include <stdint.h>

#include "kws_frontend.h"

/* ---------------------------------------------------------------- shapes --
 * All of these are derived by tools/export_m1gw_weights.py from the declared
 * config, and the exporter refuses to emit unless the checkpoint matches. They
 * are repeated here because the kernel sizes its buffers from them. */
#define KWS_M1GW_DIM        96
#define KWS_M1GW_HEADS      4
#define KWS_M1GW_HEAD_DIM   (KWS_M1GW_DIM / KWS_M1GW_HEADS)
#define KWS_M1GW_STEM       32
#define KWS_M1GW_BLOCKS     2
#define KWS_M1GW_FFN        KWS_M1GW_DIM      /* ffn_mult 1.0 */
#define KWS_M1GW_SE         12               /* max(4, 96/8)   */
#define KWS_M1GW_MEL_RANK   8
#define KWS_M1GW_CLASSES    2
#define KWS_M1GW_POOL       (3 * KWS_M1GW_DIM)   /* mean + lse + max */

#define KWS_M1GW_T_IN       KWS_FRAMES       /* 49 */
#define KWS_M1GW_M_IN       KWS_MELS         /* 40 */
#define KWS_M1GW_M1         20               /* mel after stem1, stride 2 */
#define KWS_M1GW_T2         13               /* time after stride-4 down */
#define KWS_M1GW_M2         10               /* mel after stem2 */

#define KWS_M1GW_MAX_POS    (KWS_M1GW_T_IN * KWS_M1GW_M1)   /*  980 */
#define KWS_M1GW_T2_POS     (KWS_M1GW_T2 * KWS_M1GW_M1)     /*  260 */
#define KWS_M1GW_S2_POS     (KWS_M1GW_T2 * KWS_M1GW_M2)     /*  130 */

#define KWS_M1GW_REL_BIAS_LEN 255            /* 2*max_len-1, max_len 128 */

/* ------------------------------------------------------------------ RAM --
 * One arena, sized from the shapes above, so the budget is auditable in one
 * place instead of being spread across declarations. Every buffer is int8
 * except the float scratch, which exists only where a norm, a softmax or a
 * logsumexp genuinely needs one.
 */
typedef struct __attribute__((aligned(16))) {
    /* int8 stem activations, laid out [position][channel] so every GEMM's
     * reduction axis is contiguous in both operands. */
    int8_t delta[3 * KWS_M1GW_T_IN * KWS_M1GW_M_IN];      /*  5,880 */
    int8_t s1[KWS_M1GW_STEM * KWS_M1GW_MAX_POS];          /* 31,360 */
    int8_t dn[KWS_M1GW_STEM * KWS_M1GW_T2_POS];            /*  8,320 */
    int8_t s2[KWS_M1GW_DIM * KWS_M1GW_S2_POS];            /* 12,480 */
    /* Requantised GroupNorm+SiLU output, fed straight into the next pointwise
     * GEMM. Sized for the largest stem (32 x 13 x 20). */
    int8_t gnq[KWS_M1GW_STEM * KWS_M1GW_T2_POS];           /*  8,320 */

    /* One scale per position. Each GEMM row is requantised on its own, which
     * keeps the pipeline single-pass: a single tensor-wide scale would need the
     * whole tensor computed twice to find its maximum first. 1,370 floats. */
    float s1_sc[KWS_M1GW_MAX_POS];
    float dn_sc[KWS_M1GW_T2_POS];
    float s2_sc[KWS_M1GW_S2_POS];

    /* float working sets. dwf is the depthwise + GroupNorm stage and serves all
     * three stem blocks (the largest needs 32x13x20 = 8,320). */
    float dwf[KWS_M1GW_STEM * KWS_M1GW_T2_POS];           /* 33,280 */
    float row[KWS_M1GW_DIM];                             /*    384 */
    /* dense_f32() stages each output row here before copying it back, so that
     * x and y may be the same buffer. Without it, writing y[p][o] clobbers
     * x[p][o] while o' > o still has to read it, and the result is a silent
     * NaN rather than a wrong number. Sized for the widest output (qkv, 3*DIM). */
    float dense_row[3 * KWS_M1GW_DIM];                   /*  1,152 */
#ifdef KWS_HOST_TEST
    /* Snapshot of stem1's post-norm output, for the parity harness only. */
    float snap[KWS_M1GW_MAX_POS * 3];
    float snapseq[KWS_M1GW_T2 * KWS_M1GW_DIM];
    int8_t snapq[KWS_M1GW_MAX_POS * 3];
    float snap_scale;
#endif
    float mel[KWS_M1GW_MEL_RANK * KWS_M1GW_S2_POS];      /*  4,160 */
    float seq[KWS_M1GW_T2 * KWS_M1GW_DIM];                /*  4,992 */
    float qkv[KWS_M1GW_T2 * 3 * KWS_M1GW_DIM];            /* 14,976 */
    float blk[KWS_M1GW_T2 * KWS_M1GW_DIM];                /*  4,992 */

    /* int8 GEMM staging for the conformer blocks.
     *
     * dense_f32() was scalar float32 and cost 99 ms for 1.77 M MAC -- 41% of the
     * inference, on a part with a vector int8 unit going unused. Every block
     * projection now goes through kws_accum_row_i8() instead, which needs an int8
     * input and produces an int8 output, so the blocks need these three buffers
     * between the float stages that genuinely require float (LayerNorm, softmax,
     * SiLU, the SE gate).
     *
     * `gi` is sized for the widest projection, qkv at 3*DIM. It and `gq` are
     * shared by every GEMM in the block; the int8 path needs no more live state
     * than the float path did, because the float stages reuse `blk` and `qkv`.
     */
    int8_t gq[KWS_M1GW_T2 * KWS_M1GW_DIM];              /*  1,248 */
    int8_t gi[KWS_M1GW_T2 * 3 * KWS_M1GW_DIM];          /*  3,744 */
    float gsc[KWS_M1GW_T2 * 3 * KWS_M1GW_DIM];          /*  4,992 */
    float logits[KWS_M1GW_HEADS * KWS_M1GW_T2 * KWS_M1GW_T2]; /* 2,704 */
    float pool[KWS_M1GW_POOL];                            /*  1,152 */
    float head[KWS_M1GW_CLASSES];                         /*      8 */
} kws_m1gw_arena;

/*
 * 16-byte align the whole arena.
 *
 * This is not decoration. kws_accum_row_i8() takes the Xtensa assembly path only
 * when the activation pointer AND the weight pointer are both 16-byte aligned.
 * The weight side is fixed by the exporter, but the ACTIVATION side points into
 * this struct, and a struct with only char and float members has alignment 4 --
 * so `gnq` landed at a 4-byte boundary and every stem GEMM silently fell back
 * to the scalar loop even after the blob was aligned. Measured on hardware at
 * 20.7 cycles/MAC on `down`, against a floor of well under 1.
 */
_Static_assert(__alignof__(kws_m1gw_arena) >= 16,
               "arena must be 16-byte aligned or kws_accum_row_i8() demotes to scalar");

#define KWS_M1GW_RAM_BYTES ((uint32_t)sizeof(kws_m1gw_arena))

/* The ceiling from the brief. Asserted rather than documented: a shape change
 * that quietly pushes the arena over must fail the build, not the board. */
_Static_assert(sizeof(kws_m1gw_arena) < 256u * 1024u,
               "m1_g_wide activation arena exceeds the 256 KB internal-DRAM budget");

/* The arena is not the whole cost. Every scratch buffer must live inside it:
 * a function-local `static float buf[...]` is invisible to the assertion above
 * and lands in .bss regardless. An earlier draft of this file kept the
 * dequantised stem tensors as function-local statics, which put the object at
 * 424,861 B of .bss -- 353 KB with the arena -- while the assert still passed.
 * That is the failure this comment exists to prevent. Read the depthwise stages
 * straight from the int8 tensors instead; there are now no large statics in
 * kws_m1gw.c. */

/* ------------------------------------------------------------- interface -- */

/*
 * Run the network. `spec` is [KWS_FRAMES][KWS_MELS] row-major float32, which is
 * what kws_frontend.h produces. Returns P(keyword).
 */
float kws_m1gw_run(const float *spec);

/* Inference time in microseconds for the most recent run, and the total since
 * boot. The firmware prints these; they are the only honest answer to the CPU
 * budget question, and the 10 ms detection target is measured against them. */
uint32_t kws_m1gw_last_us(void);
uint64_t kws_m1gw_total_us(void);
void     kws_m1gw_reset_timing(void);

/*
 * Per-stage cost, accumulated across runs. KWS_M1GW_STAGE_COUNT slots in
 * execution order; "the model is slow" is not an argument for a specific change,
 * and a stage costing five times its share of the MACs is a bug in that stage.
 *
 * Microseconds AND cycles are kept, because they answer different questions.
 * Cycles are what the CPU actually spent and are immune to a clock that is not
 * what you think; microseconds are what the hop budget is denominated in. On
 * this part they differ by the configured CPU frequency, and a mismatch between
 * the two is a frequency misconfiguration worth catching.
 */
/* Seven slots, one per STAGE_END() in the forward pass. It was eight, which
 * did not match the code: the conformer blocks are a single loop whose stage
 * close sat OUTSIDE it, so every microsecond of both blocks landed in the
 * 'attn' slot and the last two reported exactly zero while demonstrably
 * running. Slots are named for what they time, not for what was hoped for. */
#define KWS_M1GW_STAGE_COUNT 7
const char *kws_m1gw_stage_name(int slot);
uint64_t    kws_m1gw_stage_us(int slot);
uint64_t    kws_m1gw_stage_cycles(int slot);
uint32_t    kws_m1gw_stage_mac(int slot);
uint64_t    kws_m1gw_stage_runs(void);
void        kws_m1gw_stage_reset(void);

/* Total multiply-accumulates for one inference, walked from the declared layer
 * sequence by tools/export_m1gw_weights.py rather than scaled from a recorded
 * number. Summed across the stage table; exposed because it is the denominator
 * for "did the implementation overhead stay sane". */
uint32_t kws_m1gw_total_mac(void);

/* CPU frequency the cycle counts are denominated in, as reported by the IDF. */
uint32_t kws_m1gw_cpu_hz(void);

/*
 * Per-projection timing inside the conformer blocks.
 *
 * "blocks: 98 ms, 1771776 MAC, 13.27 cyc/MAC" is one opaque bucket, and it has
 * now misled the optimisation twice: once into converting float to int8 for no
 * gain, once into blaming libm. The arithmetic says a 96-element int8 dot should
 * cost a few cycles, and it is costing 1346, so the multiply-accumulate is not
 * where the time is -- but which of the five projections carries it is not
 * knowable from a single number.
 */
#define KWS_M1GW_PROJ_COUNT 5
const char *kws_m1gw_proj_name(int slot);
uint64_t    kws_m1gw_proj_us(int slot);
uint64_t    kws_m1gw_proj_cycles(int slot);
void        kws_m1gw_proj_reset(void);

/* Cycles and microseconds for the most recent inference, and the worst
 * microsecond figure seen since the last kws_m1gw_stage_reset(). The peak is
 * kept separately because the mean hides exactly the case that matters: a
 * single inference that overruns the hop drops samples, and a good average
 * hides it completely. */
uint32_t kws_m1gw_last_cycles(void);
uint32_t kws_m1gw_peak_us(void);

/* Bytes of internal DRAM the activation arena occupies, for the boot log. */
uint32_t kws_m1gw_ram_bytes(void);

/* The frozen operating point from the checkpoint, for Kconfig cross-checking. */
float kws_m1gw_threshold(void);

#endif /* KWS_M1GW_H */