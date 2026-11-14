# Reactant.jl front end: trace a Julia function with Reactant, take the optimized
# StableHLO module and translate it to MIL.
#
# Layout. Reactant prints a Julia array of size (d1, …, dr) as tensor<dr x … x d1>:
# row-major, the same memory. A rank-r MLIR tensor maps onto MIL rank 4 by
# prepending ones ([n] -> [1,1,1,n], [a,b] -> [1,a,1,b], [a,b,c] -> [1,a,b,c]),
# which keeps row-major order, so reshape, slice and concatenate translate axis
# by axis. A Julia matrix thus lands in the [1, C, 1, S] layout, as with the
# built-in tracer. Computation is in fp16.
module AppleNeuralEngineReactantExt

using AppleNeuralEngine
using AppleNeuralEngine: MIL
using Reactant

const MLIR = Reactant.MLIR
const IR = MLIR.IR

_unsupported(what) = throw(ArgumentError("frontend=:reactant does not support $what"))

# --- layout ----------------------------------------------------------------------

function _m4(s)
    r = length(s)
    r == 0 && return [1, 1, 1, 1]
    r == 1 && return [1, 1, 1, s[1]]
    r == 2 && return [1, s[1], 1, s[2]]
    r == 3 && return [1, s[1], s[2], s[3]]
    _unsupported("rank-$r tensors")
end
# MLIR dimension d (0-based) of a rank-r tensor -> MIL axis (0-based)
_axis(r, d) = (r == 1 ? (3,) : r == 2 ? (1, 3) : (1, 2, 3))[d+1]

_shape(t::IR.Type) = Int[size(t)...]
_shape(v::IR.Value) = _shape(IR.type(v))
_eltype(v::IR.Value) = IR.julia_type(eltype(IR.type(v)))

# --- attributes, read from their printed form -------------------------------------

_attr(op, name) = (a = IR.getattr(op, name); a === nothing ? nothing : string(a))
# "array<i64: 1, 2>" -> [1, 2]
_ints(s) = s === nothing ? Int[] : parse.(Int, getfield.(eachmatch(r"-?\d+", split(s, ':'; limit=2)[end]), :match))
# "0 : i64" -> 0
_int(s) = parse(Int, match(r"-?\d+", s).match)
function _dotdims(op)
    s = _attr(op, "dot_dimension_numbers")
    grab(key) = (m = match(Regex(key * raw" = \[([\d, ]*)\]"), s);
                 m === nothing ? Int[] : parse.(Int, split(m[1], ','; keepempty=false)))
    return (lb=grab("lhs_batching_dimensions"), rb=grab("rhs_batching_dimensions"),
            lc=grab("lhs_contracting_dimensions"), rc=grab("rhs_contracting_dimensions"))
end

# --- values ------------------------------------------------------------------------

# A MIL variable (possibly not yet broadcast to its full shape) or a scalar from a
# splat constant, which MIL broadcasts for free.
struct Val
    x::Union{MIL.Var,Float64}
    shape::Vector{Int}          # MLIR shape
end

_scalar(x) = x isa Float64 ? Float16(x) : x
_key(v::IR.Value) = v.ref.ptr

# Materialize `v` at its full MIL rank-4 shape.
function _full(g, v::Val)
    target = _m4(v.shape)
    v.x isa Float64 && return MIL.constant(g, fill(Float16(v.x), target...))
    v.x.shape == target && return v.x
    return MIL.tile(v.x, target .÷ v.x.shape)
end

# Julia array in MLIR row-major order -> MIL constant in the rank-4 layout.
_constant(g, vals, shape) = MIL.constant(g, permutedims(reshape(Float16.(vals), reverse(_m4(shape))...), (4, 3, 2, 1)))

const _UNARY = Dict(
    "stablehlo.tanh" => (MIL.tanh, tanh), "stablehlo.exponential" => (MIL.exp, exp),
    "stablehlo.sqrt" => (MIL.sqrt, sqrt), "stablehlo.abs" => (MIL.abs, abs),
    "stablehlo.logistic" => (MIL.sigmoid, x -> 1 / (1 + exp(-x))), "stablehlo.floor" => (MIL.floor, floor),
    "stablehlo.ceil" => (MIL.ceil, ceil), "stablehlo.sine" => (MIL.sin, sin), "stablehlo.cosine" => (MIL.cos, cos),
    "stablehlo.sign" => (MIL.sign, sign), "stablehlo.round_nearest_even" => (MIL.round, round),
    "stablehlo.log" => (x -> MIL.log(x), log), "stablehlo.rsqrt" => (x -> MIL.rsqrt(x), x -> 1 / sqrt(x)),
    "stablehlo.negate" => (x -> MIL.mul(x, Float16(-1)), -))
const _BINARY = Dict(
    "stablehlo.add" => (MIL.add, +), "stablehlo.subtract" => (MIL.sub, -),
    "stablehlo.multiply" => (MIL.mul, *), "stablehlo.divide" => (MIL.real_div, /),
    "stablehlo.maximum" => (MIL.maximum, max), "stablehlo.minimum" => (MIL.minimum, min),
    "stablehlo.power" => (MIL.pow, ^))
const _COMPARE = Dict("EQ" => MIL.equal, "NE" => MIL.not_equal, "GT" => MIL.greater,
                      "GE" => MIL.greater_equal, "LT" => MIL.less, "LE" => MIL.less_equal)
const _REDUCE = Dict("stablehlo.add" => (MIL.reduce_sum, 0.0), "stablehlo.maximum" => (MIL.reduce_max, -Inf),
                     "stablehlo.minimum" => (MIL.reduce_min, Inf), "stablehlo.multiply" => (MIL.reduce_prod, 1.0))

# Per-trace state: where each slice came from (for `_reslice`), and the
# channel-major copy of each tensor that offset slices are taken from.
struct _Ctx
    slices::Dict{String,Tuple{MIL.Var,Vector{Int},Vector{Int}}}
    flipped::Dict{String,MIL.Var}
end
_Ctx() = _Ctx(Dict{String,Tuple{MIL.Var,Vector{Int},Vector{Int}}}(), Dict{String,MIL.Var}())

# x[b:e:s] in MIL axes. On the ANE, a slice along the innermost (width) axis that
# starts at some nonzero offsets, or is strided, runs through the arithmetic units:
# exact only for |x| in [1, 4096) (M2, macOS 27). Slices along the other axes, and
# transposes and reshapes, are plain copies. So take such a slice from a copy of x
# with width and channels swapped (by reshape for vectors), and swap back.
function _slice(ctx, x::MIL.Var, b, e, s)
    stride = all(==(1), s) ? nothing : s
    b[4] == 0 && s[4] == 1 && return MIL.slice_by_index(x, b, e; stride)
    # First the other axes, directly (keeping the full width), then the width alone,
    # so that no offset lands on the innermost axis of the swapped copy.
    if any(i -> b[i] != 0 || e[i] != x.shape[i] || s[i] != 1, 1:3)
        w = x.shape[4]
        x = MIL.slice_by_index(x, [b[1:3]; 0], [e[1:3]; w]; stride=all(==(1), s[1:3]) ? nothing : [s[1:3]; 1])
        b, e, s = [0, 0, 0, b[4]], [x.shape[1:3]; e[4]], [1, 1, 1, s[4]]
        stride = s[4] == 1 ? nothing : s
    end
    perm = [1, 4, 3, 2]                                      # swap axes 1 and 3 (1-based)
    vector = x.shape[2] == 1 && x.shape[3] == 1
    t = get!(ctx.flipped, x.name) do
        vector ? MIL.reshape(x, x.shape[perm]) : MIL.transpose(x, perm .- 1)
    end
    y = MIL.slice_by_index(t, b[perm], e[perm]; stride=stride === nothing ? nothing : s[perm])
    return vector ? MIL.reshape(y, y.shape[perm]) : MIL.transpose(y, perm .- 1)
end

function _convert!(g, env, ctx, op)
    name = IR.name(op)
    arg(i) = env[_key(IR.operand(op, i))]
    rshape = IR.nresults(op) > 0 ? _shape(IR.result(op, 1)) : Int[]
    set!(x) = (env[_key(IR.result(op, 1))] = Val(x, rshape))

    if name == "stablehlo.constant"
        a = IR.getattr(op, "value")
        T = IR.julia_type(eltype(IR.type(a)))
        T in (Float16, Float32, Float64, Int32, Int64) || _unsupported("constants of element type $T")
        splat = MLIR.API.mlirDenseElementsAttrIsSplat(a)
        n = splat ? 1 : prod(rshape; init=1)
        raw = Ptr{T}(MLIR.API.mlirDenseElementsAttrGetRawData(a))
        vals = [unsafe_load(raw, i) for i in 1:n]
        set!(splat ? Float64(vals[1]) : _constant(g, vals, rshape))
    elseif haskey(_UNARY, name)
        x = arg(1)
        mf, sf = _UNARY[name]
        set!(x.x isa Float64 ? Float64(sf(x.x)) : mf(x.x))
    elseif haskey(_BINARY, name)
        a, b = arg(1), arg(2)
        mf, sf = _BINARY[name]
        set!(a.x isa Float64 && b.x isa Float64 ? Float64(sf(a.x, b.x)) : mf(_scalar(a.x), _scalar(b.x)))
    elseif name == "stablehlo.compare"
        dir = match(r"comparison_direction\s+(\w+)", _attr(op, "comparison_direction"))[1]
        set!(_COMPARE[dir](_scalar(arg(1).x), _scalar(arg(2).x)))
    elseif name == "stablehlo.select"
        c, a, b = arg(1), arg(2), arg(3)
        c.x isa MIL.Var || _unsupported("select with a constant condition")
        set!(MIL.select(c.x, _scalar(a.x), _scalar(b.x)))
    elseif name == "stablehlo.convert"
        x = arg(1)
        # Bool -> number; float <-> float is a no-op, since computation stays fp16.
        set!(x.x isa MIL.Var && x.x.eltype === Bool ? MIL.select(x.x, Float16(1), Float16(0)) : x.x)
    elseif name == "stablehlo.broadcast_in_dim"
        x = arg(1)
        if x.x isa Float64
            set!(x.x)
        else
            inter = ones(Int, length(rshape))
            for (i, d) in enumerate(_ints(_attr(op, "broadcast_dimensions")))
                inter[d+1] = x.shape[i]
            end
            # Tiled lazily by `_full`: elementwise consumers broadcast it for free.
            set!(x.x.shape == _m4(inter) ? x.x : MIL.reshape(_full(g, x), _m4(inter)))
        end
    elseif name == "stablehlo.reshape"
        set!(MIL.reshape(_full(g, arg(1)), _m4(rshape)))
    elseif name == "stablehlo.transpose"
        x = arg(1); r = length(x.shape)
        perm = collect(0:3)
        for (i, d) in enumerate(_ints(_attr(op, "permutation")))
            perm[_axis(r, i - 1)+1] = _axis(r, d)
        end
        set!(MIL.transpose(_full(g, x), perm))
    elseif name == "stablehlo.slice"
        x = arg(1); r = length(x.shape)
        st, li, sd = _ints(_attr(op, "start_indices")), _ints(_attr(op, "limit_indices")), _ints(_attr(op, "strides"))
        isempty(sd) && (sd = ones(Int, r))
        b, e, s = zeros(Int, 4), _m4(x.shape), ones(Int, 4)
        for d in 0:r-1
            a = _axis(r, d) + 1
            b[a], e[a], s[a] = st[d+1], li[d+1], sd[d+1]
        end
        src = _full(g, x)
        y = _slice(ctx, src, b, e, s)
        all(==(1), s) && (ctx.slices[y.name] = (src, b, e))      # provenance, for concatenate
        set!(y)
    elseif name == "stablehlo.concatenate"
        xs = [_full(g, arg(i)) for i in 1:IR.noperands(op)]
        axis = _axis(length(rshape), _int(_attr(op, "dimension")))
        set!(MIL.concat(_reslice(xs, ctx.slices); axis))
    elseif name == "stablehlo.pad"
        x, pv = arg(1), arg(2); r = length(x.shape)
        pv.x isa Float64 || _unsupported("pad with a computed padding value")
        lo, hi = _ints(_attr(op, "edge_padding_low")), _ints(_attr(op, "edge_padding_high"))
        any(!=(0), _ints(_attr(op, "interior_padding"))) && _unsupported("interior padding")
        (any(<(0), lo) || any(<(0), hi)) && _unsupported("negative padding")
        # MIL's constant `pad` runs through the arithmetic units on the ANE and loses
        # values below 2^-7 in magnitude; concatenating constant blocks copies exactly.
        y = _full(g, x)
        for d in 0:r-1
            a = _axis(r, d) + 1
            parts = MIL.Var[]
            for (n, last) in ((lo[d+1], false), (hi[d+1], true))
                n == 0 && continue
                shape = copy(y.shape); shape[a] = n
                block = MIL.constant(g, fill(Float16(pv.x), shape...))
                last ? push!(parts, block) : pushfirst!(parts, block)
            end
            isempty(parts) && continue
            pieces = lo[d+1] > 0 ? [parts[1], y, parts[2:end]...] : [y, parts...]
            y = MIL.concat(pieces; axis=a - 1)
        end
        set!(y)
    elseif name == "stablehlo.iota"
        d = _int(_attr(op, "iota_dimension"))
        A = Array{Float16}(undef, reverse(rshape)...)       # Julia view of the MLIR tensor
        for I in CartesianIndices(A)
            A[I] = I[length(rshape)-d] - 1
        end
        set!(_constant(g, vec(A), rshape))
    elseif name == "stablehlo.dot_general"
        a, b = arg(1), arg(2)
        dd = _dotdims(op)
        isempty(dd.lb) && isempty(dd.rb) || _unsupported("batched dot_general")
        length(dd.lc) == 1 && length(dd.rc) == 1 || _unsupported("dot_general with several contracting dimensions")
        ra, rb = length(a.shape), length(b.shape)
        ra <= 2 && rb <= 2 || _unsupported("dot_general on rank > 2")
        # As matrices: lhs [M,K] (a vector [K] is [1,K]); rhs [K,N] (a vector [K] is [K,1]).
        A = MIL.reshape(_full(g, a), ra == 1 ? [1, a.shape[1]] : a.shape)
        B = MIL.reshape(_full(g, b), rb == 1 ? [b.shape[1], 1] : b.shape)
        y = MIL.matmul(A, B; transpose_x=ra == 2 && dd.lc[1] == 0, transpose_y=rb == 2 && dd.rc[1] == 1)
        set!(MIL.reshape(y, _m4(rshape)))
    elseif name == "stablehlo.reduce"
        IR.noperands(op) == 2 || _unsupported("variadic reduce")
        x, init = arg(1), arg(2)
        body = IR.name(first(IR.first_block(IR.region(op, 1))))
        haskey(_REDUCE, body) || _unsupported("reduce with body $body")
        f, identity = _REDUCE[body]
        r = length(x.shape)
        y = f(_full(g, x); axes=[_axis(r, d) for d in _ints(_attr(op, "dimensions"))], keep_dims=true)
        y = MIL.reshape(y, _m4(rshape))
        init.x isa Float64 && init.x == identity || (y = _BINARY[body][1](y, _scalar(init.x)))
        set!(y)
    else
        _unsupported("the StableHLO op $name")
    end
end

# The ANE compiler mishandles a `concat` of two slices of the same tensor that start
# at the same index (M2, macOS 27): depending on the sizes it fails ("Dimensions of
# programmed input and output must match") or silently computes wrong values.
# Reactant emits exactly that for an edge-replicating shift, `vcat(x[1:1], x[1:end-1])`.
# Take the shorter slice from the longer one instead, which is exact and fast.
function _reslice(xs, slices)
    out = collect(xs)
    for i in eachindex(out), j in eachindex(out)
        i == j && continue
        out[i] === xs[i] || continue                              # already re-sliced
        si, sj = get(slices, xs[i].name, nothing), get(slices, xs[j].name, nothing)
        (si === nothing || sj === nothing) && continue
        (srci, bi, ei), (srcj, bj, ej) = si, sj
        if srci.name == srcj.name && bi == bj && all(ei .<= ej) && ei != ej
            out[i] = MIL.slice_by_index(xs[j], zeros(Int, 4), ei .- bj)
        end
    end
    return out
end

function AppleNeuralEngine._reactant_trace(f, inputs...)
    arrays = map(inputs) do spec
        A = spec isa AbstractArray ? spec : zeros(Float16, Tuple(spec)...)
        eltype(A) in (Float16, Float32) || throw(ArgumentError("inputs must be Float16 or Float32 arrays, got $(eltype(A))"))
        ndims(A) <= 3 || throw(ArgumentError("frontend=:reactant supports arrays of up to 3 dimensions"))
        A
    end
    rarrays = map(Reactant.to_rarray, arrays)
    IR.@dispose ctx = Reactant.ReactantContext() begin
        mod = Reactant.Compiler.code_hlo(ctx, f, rarrays)
        try
            fns = [op for op in IR.body(mod) if IR.name(op) == "func.func"]
            length(fns) == 1 || _unsupported("modules with several functions")
            block = IR.first_block(IR.region(only(fns), 1))
            g = MIL.Graph()
            env = Dict{Any,Val}()
            state = _Ctx()
            # Declare every input before converting any op: MIL inputs precede all ops.
            vars = map(1:IR.nargs(block)) do i
                v = IR.argument(block, i)
                s = _shape(v)
                (MIL.input(g, "x$i", _eltype(v), _m4(s); julia_shape=Tuple(reverse(s))), v, s)
            end
            for (x, v, s) in vars
                env[_key(v)] = Val(x.eltype == Float16 ? x : MIL.cast(x, Float16), s)
            end
            for op in block
                if IR.name(op) in ("func.return", "stablehlo.return")
                    for i in 1:IR.noperands(op)
                        v = IR.operand(op, i)
                        val = env[_key(v)]
                        x = _full(g, val)
                        x.eltype === Bool && (x = MIL.select(x, Float16(1), Float16(0)))
                        # A program output must be produced by an op.
                        any(inp -> inp.name == x.name, g.inputs) && (x = MIL.mul(x, Float16(1)))
                        T = _eltype(v) === Float32 ? Float32 : Float16
                        MIL.output!(g, T === Float16 ? x : MIL.cast(x, T), "y$i"; julia_shape=Tuple(reverse(val.shape)))
                    end
                elseif IR.name(op) == "func.call"
                    _unsupported("function calls (func.call)")
                else
                    _convert!(g, env, state, op)
                end
            end
            return g
        finally
            IR.dispose(mod)
        end
    end
end

end # module
