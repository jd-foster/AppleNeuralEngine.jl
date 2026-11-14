# Hardware notes

Observations made while building and testing this package on an M2 Max
(16-core ANE, macOS 27). They complement the restrictions catalogued by
[Orion (arXiv 2603.06728)](https://arxiv.org/abs/2603.06728).

## Precision

- The ANE computes in **fp16**. Ops on fp32 tensors do not compile, and neither
  does a cast to fp32 in the middle of a graph. Only program inputs and outputs
  may be fp32.
- `round` rounds ties away from zero, computed as `sign(x)·floor(|x| + 0.5)` in
  fp16. For `|x|` just below a half-integer the addition can round up.
- `floor_div` uses an approximate reciprocal and can be off by one when the
  quotient is (nearly) an integer. `atan` is accurate to about 2.5%.
- `pad` with `mode="constant"` does not copy values exactly: `|x| < 2^-7` snaps to
  multiples of 2^-17 (subnormals become 0), the `[2^-7, 2^-6)` binade is off by up
  to 1 ulp, and `|x| ≥ 2^15` becomes ±Inf. `mode="reflect"` is exact.
  `mode="replicate"` is as lossy as `"constant"`, and is rejected on the channel axis.
- Data movement is exact only when the compiler lowers it to plain copies.
  Convolutions and matrix products with 0/1 weights flush tiny values like any
  arithmetic. A `slice_by_index` along the innermost (width) axis that starts at
  certain nonzero offsets (1–14 and 100 among those tried, but not 15–17, 24, 32 or 48), or
  `reverse` along it, is exact only for `|x|` in [1, 4096): smaller values lose up to
  3% (below 2^-6 they become 0) and larger ones saturate. Slices along the other
  axes, `transpose` and `reshape` are exact. To take an offset slice along the width
  exactly, swap width and channels (`transpose`, or `reshape` for a `[1, 1, 1, N]`
  vector), slice, and swap back; this costs about as much as the direct slice.

## Buffers

- **Row alignment.** Tensor buffers are 2-D IOSurfaces with one row per run of
  the innermost MIL dimension. The ANE rejects rows that are not 64-byte aligned
  (16 fp32 or 32 fp16 values). The runtime reports this as an evaluation failure.
  With the row *stride* padded to 64 bytes, any innermost size works, down to
  1. This package does that, so the "pad the sequence dimension to ≥ 16" rule
  reported elsewhere does not apply.
- **Buffer sizes** must match what the compiled model expects. The runtime reports
  "IOSurface smaller than the model expects" otherwise.

## Program structure

- Inputs and outputs are bound in sorted-name order (see
  [Raw MIL and the runtime](@ref)).
- A program output cannot be produced by a trailing `identity` op.
- Variable names must not be MIL keywords (`const`, `tensor`, type names).
- A `concat` of two slices of the same tensor that start at the same index, such
  as `concat([x[0:1], x[0:N-1]])` (an edge-replicating shift), is mishandled:
  depending on the sizes it fails to compile ("Dimensions of programmed input and
  output must match", or "Dimension of input tensor #1 are not broadcastable" when
  an elementwise op consumes it) or **silently returns wrong values**. Inserting
  `identity` or a multiplication by 1 does not help; the compiler removes them.
  Take the shorter slice from the longer one instead, `y = x[0:N-1]` then
  `concat([y[0:1], y])`, which is exact.

## Cost

- Compiling and loading a program takes about 20 ms for a tiny graph and about
  70–80 ms for a transformer block. We ran 300 compile/run/close cycles in one
  process without hitting a compile limit, as long as programs are closed.
- Each evaluation costs about 0.1–0.2 ms on top of the compute, for the call and
  the buffer copies. Small programs are therefore latency-bound. Batch work into
  larger matrix-matrix products.
- Allocating a fresh output array per call makes timings erratic for large outputs
  (page faults on new memory): for a 2048×2048 product the median call took about
  twice the minimum. Reuse outputs with `evaluate!(outputs, program, inputs...)`,
  which allocates nothing.
- Under sustained load the ANE slows down: after about half a second of back-to-back
  calls, a 2048×2048 product settled about 50% slower than in the first burst
  (median 1.75 ms, then about 2.6 ms, on an M2 Max). Short bursts and long runs
  therefore give different numbers; `benchmark/run.jl` measures bursts by default.
- Calling the runtime through ObjectiveC.jl instead of a compiled Objective-C
  bridge adds about 0.1 µs per call, which is not measurable against the ANE
  itself.
