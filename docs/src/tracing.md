# Tracing Julia functions

`compile(f, inputs...)` runs `f` once on placeholder arrays, records every
operation as a MIL graph, and compiles the graph for the Neural Engine. The same
function keeps working on ordinary arrays, so you can check the ANE result
against the CPU directly.

```julia
using AppleNeuralEngine

const EPS = 1f-5
silu(z) = z / (1 + exp(-z))

# A LLaMA-style feed-forward block: RMSNorm, SwiGLU, residual.
function ffn(x)
    n = x ./ sqrt.(sum(abs2, x; dims=2) ./ size(x, 2) .+ EPS) .* g'
    return x .+ (silu.(n * W1') .* (n * W3')) * W2'
end

x = randn(Float16, 256, 512)
program = compile(ffn, x)        # inputs: example arrays (or sizes)
y = program(x)                   # on the ANE
y_cpu = ffn(Float32.(x))         # the same function on the CPU
```

## What is supported

Inputs are vectors or matrices of `Float16` or `Float32`. The output has the
same element type as the first input. Inside the trace everything is computed
in fp16.

- **Broadcasting** with any scalar function built from `+ - * / ^`, `exp`,
  `exp2`, `log`, `sqrt`, `inv`, `abs`, `abs2`, `tanh`, `max`, `min`, `floor`,
  `ceil`, `round`, `sign`, comparisons and `ifelse`. Whole expressions such as
  `silu.(x)` or `ifelse.(x .> 0, x, 0.1f0 .* x)` are traced through the scalar
  function.
- **Matrix products** `A * B` between traced arrays and constant matrices,
  `adjoint`/`transpose`, and matrix-vector products.
- **Reductions** `sum`, `maximum`, `minimum`, with or without `dims`, and
  `sum(f, x; dims)`.
- **Constants:** any Julia array captured by the function (weights, biases) is
  stored in the program's weight file. An array used several times is stored
  once.
- **Several outputs:** return a tuple.

`x * W` with a constant `W` becomes a 1×1 convolution, the Neural Engine's
fastest operation, so write layers in that orientation: rows are tokens or
samples, columns are features.

## The Reactant front end

The built-in tracer covers vectors, matrices and a fixed set of operations. For
anything more, `frontend=:reactant` traces the function with
[Reactant.jl](https://github.com/EnzymeAD/Reactant.jl), lets Reactant optimize it,
and translates the resulting StableHLO module to MIL. Reactant is an optional
dependency: the front end becomes available after `using Reactant`.

```julia
using AppleNeuralEngine, Reactant

function smooth(x)                  # indexing, vcat and a loop: not traceable by the built-in tracer
    for _ in 1:16
        x = 0.5f0 .* x .+ 0.25f0 .* (vcat(x[2:end], x[end:end]) .+ vcat(x[1:1], x[1:end-1]))
    end
    return x
end

x = rand(Float32, 256)
program = compile(smooth, x; frontend=:reactant)
program(x) ≈ smooth(x)              # up to fp16 rounding
```

Loops with a trip count known at trace time are unrolled, so the whole loop runs
as one ANE program (a `while` loop would not run on the ANE). Supported: elementwise
arithmetic and functions, comparisons and `ifelse`, broadcasting, matrix products
(`dot_general` without batch dimensions), reductions (`sum`, `maximum`, `minimum`),
reshapes, transposes, slices, concatenation, padding and `iota`, on arrays of up to
3 dimensions. Anything else raises an `ArgumentError` naming the StableHLO op.

Indexing, shifts, concatenation and padding copy values exactly, tiny and huge ones
included: the front end avoids the ANE's lossy paths for these and works around an
ANE compiler bug (see [Hardware notes](@ref)). Data movement still costs more than
arithmetic: an unrolled 3-point stencil on 256 cells costs about 0.6 µs per step
(1.1 µs on 64 columns of 256), against 0.15 µs per step when the same stencil is
written as a convolution with [The MIL graph builder](@ref).

This front end is experimental: it relies on Reactant's internal MLIR interface,
which can change between Reactant releases.

## Layout

A traced `(T, D)` matrix is held as the MIL tensor `[1, D, 1, T]`. That is the
same memory as the Julia matrix and the layout the ANE is fastest with, so
inputs and outputs need no transposition.

## Limitations

- Only vectors and matrices; no indexing, `getindex`, `reshape` or mutation.
- `sin`, `cos` and `tan` are rejected by the ANE compiler (see [Op support](@ref)).
- Control flow must not depend on array values. The trace records one path.
- For anything else, drop to [The MIL graph builder](@ref): [`trace`](@ref)
  returns the graph, which you can extend before compiling.
