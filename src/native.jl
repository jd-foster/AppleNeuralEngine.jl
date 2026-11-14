# Direct bindings to the private AppleNeuralEngine framework, written with
# ObjectiveC.jl. The call sequence (in-memory MIL model -> compile -> load ->
# IOSurface-backed request -> evaluate) follows the reverse-engineering work in
# maderix/ANE (https://github.com/maderix/ANE).

const _ANE_FRAMEWORK = "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine"
const _IOSURFACE = "/System/Library/Frameworks/IOSurface.framework/IOSurface"
const _ANE_CLASSES = ("_ANEInMemoryModelDescriptor", "_ANEInMemoryModel", "_ANERequest", "_ANEIOSurfaceObject")
# QOS_CLASS_DEFAULT; the value used by maderix/ANE for every runtime call.
const _QOS = Cuint(0x15)
const _INITIALIZED = Ref(false)
# An empty options dictionary shared by every evaluation, so that `evaluate!`
# allocates nothing. Created by `initialize!` and never released.
const _NO_OPTIONS = Ref{id{NSDictionary}}(nil)

# Managed wrappers: `@objc [...]::T` retains the result and a finalizer releases
# it. `ANEProgram` keeps its model and request as raw pointers with one explicit
# retain each (see `_retain`), so that the program's own finalizer can unload the
# model before anything releases it: Julia does not order finalizers.
@objcwrapper _ANEInMemoryModelDescriptor <: NSObject
@objcwrapper _ANEInMemoryModel <: NSObject
@objcwrapper _ANERequest <: NSObject
@objcwrapper _ANEIOSurfaceObject <: NSObject

"""
    ANEError(msg)

Raised when the ANE runtime rejects a model or fails to run it. `msg` includes
the runtime's own `NSError` description when one is available.
"""
struct ANEError <: Exception
    msg::String
end

Base.showerror(io::IO, err::ANEError) = print(io, "ANEError: ", err.msg)

function _throw_ane(what, err::Out{NSError})
    e = err[]
    detail = e === nothing ? "unknown error" : String(@objc [e::id{NSError} description]::NSString)
    throw(ANEError("$what: $detail"))
end

"""
    initialize!()

Load the private AppleNeuralEngine framework and check that the classes this
package relies on exist. Called automatically by [`compile`](@ref).
"""
function initialize!()
    _INITIALIZED[] && return nothing
    Sys.isapple() || error("AppleNeuralEngine requires macOS")
    Sys.ARCH === :aarch64 || error("AppleNeuralEngine requires Apple Silicon (arm64)")
    isnothing(Libdl.dlopen(_ANE_FRAMEWORK; throw_error=false)) &&
        throw(ANEError("could not load AppleNeuralEngine.framework"))
    for name in _ANE_CLASSES
        ccall(:objc_getClass, Ptr{Cvoid}, (Cstring,), name) == C_NULL &&
            throw(ANEError("private class $name is missing; this macOS release may have changed the ANE runtime"))
    end
    _NO_OPTIONS[] = _retain(NSDictionary())
    _INITIALIZED[] = true
    return nothing
end

# A raw, owned (+1) pointer to `obj`, for `ANEProgram`; balanced by `_release`.
_retain(obj::T) where {T<:Object} = ccall(:objc_retain, id{T}, (id{T},), obj)
_release(obj::id) = ccall(:objc_release, Cvoid, (id{Object},), obj)

_nsdata(bytes::Vector{UInt8}) =
    @objc [NSData dataWithBytes:bytes::Ptr{UInt8} length:length(bytes)::NSUInteger]::NSData

# Foundation collections from heterogeneous objects (ObjectiveC.jl's
# constructors need homogeneous element types). The collections retain their
# elements; the returned wrappers are managed.
function _nsdict(pairs)
    dict = @objc [NSMutableDictionary dictionary]::NSDictionary
    for (key, value) in pairs
        @objc [dict::id{NSDictionary} setObject:value::id{Object} forKey:key::id{Object}]::Nothing
    end
    return dict
end

function _nsarray(items)
    array = @objc [NSMutableArray array]::NSArray
    for item in items
        @objc [array::id{NSArray} addObject:item::id{Object}]::Nothing
    end
    return array
end

# IOSurface property keys are global constant strings: borrowing them is safe.
function _surface_key(name::Symbol)
    lib = Libdl.dlopen(_IOSURFACE)
    return NSString(unsafe_load(Ptr{id{NSString}}(Libdl.dlsym(lib, name))))
end

# Tensor buffers are 2-D IOSurfaces: one row per run of the innermost
# dimension, with the row stride padded to 64 bytes. The ANE rejects buffers
# whose rows are not 64-byte aligned; with the padded stride any innermost size
# works, down to a single element.
const _ROW_ALIGN = 64

function _create_surface((rowbytes, rows))
    stride = cld(rowbytes, _ROW_ALIGN) * _ROW_ALIGN
    props = _nsdict((
        _surface_key(:kIOSurfaceWidth) => NSNumber(Int(rowbytes)),
        _surface_key(:kIOSurfaceHeight) => NSNumber(Int(rows)),
        _surface_key(:kIOSurfaceBytesPerElement) => NSNumber(1),
        _surface_key(:kIOSurfaceBytesPerRow) => NSNumber(Int(stride)),
        _surface_key(:kIOSurfaceAllocSize) => NSNumber(Int(stride * rows)),
        _surface_key(:kIOSurfacePixelFormat) => NSNumber(0),
    ))
    surface = ccall((:IOSurfaceCreate, _IOSURFACE), Ptr{Cvoid}, (id{NSDictionary},), props)
    surface == C_NULL && throw(ANEError("could not allocate a $rows × $rowbytes-byte IOSurface"))
    return surface
end

_free_surface(surface::Ptr{Cvoid}) = ccall(:CFRelease, Cvoid, (Ptr{Cvoid},), surface)

const _LOCK_READONLY = UInt32(1)

# Copy between a dense Julia buffer (rows of `rowbytes`) and a surface, honoring
# the surface's row stride.
function _copy_surface!(surface::Ptr{Cvoid}, buffer::Ptr, rowbytes::Integer, rows::Integer, to_surface::Bool)
    lock = to_surface ? UInt32(0) : _LOCK_READONLY
    ccall((:IOSurfaceLock, _IOSURFACE), Cint, (Ptr{Cvoid}, UInt32, Ptr{UInt32}), surface, lock, C_NULL)
    base = ccall((:IOSurfaceGetBaseAddress, _IOSURFACE), Ptr{UInt8}, (Ptr{Cvoid},), surface)
    stride = Int(ccall((:IOSurfaceGetBytesPerRow, _IOSURFACE), Csize_t, (Ptr{Cvoid},), surface))
    buffer = Ptr{UInt8}(buffer)
    if stride == rowbytes
        to_surface ? unsafe_copyto!(base, buffer, rowbytes * rows) : unsafe_copyto!(buffer, base, rowbytes * rows)
    else
        for r in 0:rows-1
            to_surface ? unsafe_copyto!(base + r * stride, buffer + r * rowbytes, rowbytes) :
                         unsafe_copyto!(buffer + r * rowbytes, base + r * stride, rowbytes)
        end
    end
    ccall((:IOSurfaceUnlock, _IOSURFACE), Cint, (Ptr{Cvoid}, UInt32, Ptr{UInt32}), surface, lock, C_NULL)
    return nothing
end

# The runtime compiles from a directory named after the model's identifier in
# NSTemporaryDirectory(); populate it with the MIL text and weight files.
function _populate_model_dir(model::_ANEInMemoryModel, mil, weights)
    hex = String(@objc [model::id{_ANEInMemoryModel} hexStringIdentifier]::NSString)
    # NSTemporaryDirectory() returns an autoreleased string: copy it inside a pool.
    tmp = @unsafe_autoreleasepool String(NSString(ccall(:NSTemporaryDirectory, id{NSString}, ())))
    dir = joinpath(tmp, hex)
    mkpath(joinpath(dir, "weights"))
    write(joinpath(dir, "model.mil"), mil)
    for (path, blob) in weights
        file = joinpath(dir, chopprefix(path, "@model_path/"))
        mkpath(dirname(file))
        write(file, blob)
    end
    return dir
end

function _load!(model::_ANEInMemoryModel, err::Out{NSError})
    @objc [model::id{_ANEInMemoryModel} loadWithQoS:_QOS::Cuint
                                        options:_NO_OPTIONS[]::id{NSDictionary}
                                        error:err::Out{NSError}]::Bool
end

# Compile and load `mil` with the given weight files, allocate one IOSurface per
# tensor and build the request binding them. Returns everything `ANEProgram`
# owns: (model, request, input surfaces, output surfaces, model directory).
# `inlayouts`/`outlayouts` are (row bytes, rows) per tensor.
function _compile_native(mil::Vector{UInt8}, weights, inlayouts, outlayouts)
    # Every wrapper made here is released eagerly on the way out (`release` is
    # idempotent and safe before the finalizer). Otherwise each would hold its
    # object until the next garbage collection, and the ANE only reclaims a
    # model's resources when the model is deallocated: compiling in a loop would
    # run out ("no ANE resources") long before the collector runs.
    temps = Object[]
    keep(x) = (x === nothing || push!(temps, x); x)
    try
        entries = map(weights) do (path, blob)
            entry = keep(_nsdict((keep(NSString("offset")) => keep(NSNumber(0)),
                                  keep(NSString("data")) => keep(_nsdata(blob)))))
            keep(NSString(path)) => entry
        end
        wdict = keep(_nsdict(entries))
        desc = keep(@objc [_ANEInMemoryModelDescriptor modelWithMILText:keep(_nsdata(mil))::id{NSData}
                                                       weights:wdict::id{NSDictionary}
                                                       optionsPlist:nil::id{NSDictionary}]::Union{Nothing,_ANEInMemoryModelDescriptor})
        desc === nothing && throw(ANEError("the runtime rejected the MIL text or weights"))
        model = keep(@objc [_ANEInMemoryModel inMemoryModelWithDescriptor:desc::id{_ANEInMemoryModelDescriptor}]::Union{Nothing,_ANEInMemoryModel})
        model === nothing && throw(ANEError("could not create an in-memory model"))

        dir = _populate_model_dir(model, mil, weights)
        try
            err = Out{NSError}()
            @objc([model::id{_ANEInMemoryModel} compileWithQoS:_QOS::Cuint
                                                options:_NO_OPTIONS[]::id{NSDictionary}
                                                error:err::Out{NSError}]::Bool) ||
                _throw_ane("ANE compilation failed", err)
            # Retry once: the ANE may need a moment to reclaim a slot from a
            # recently unloaded model. (No autorelease pool is open here, so the
            # task may yield.)
            if !_load!(model, Out{NSError}())
                sleep(0.1)
                err = Out{NSError}()
                _load!(model, err) || _throw_ane("ANE load failed", err)
            end
        catch
            rm(dir; recursive=true, force=true)
            rethrow()
        end

        inputs = map(_create_surface, inlayouts)
        outputs = map(_create_surface, outlayouts)
        wrap(surface) = keep(@objc [_ANEIOSurfaceObject objectWithIOSurface:surface::Ptr{Cvoid}]::_ANEIOSurfaceObject)
        indices(n) = keep(_nsarray(map(i -> keep(NSNumber(i)), 0:n-1)))
        request = keep(@objc [_ANERequest requestWithInputs:keep(_nsarray(map(wrap, inputs)))::id{NSArray}
                                          inputIndices:indices(length(inputs))::id{NSArray}
                                          outputs:keep(_nsarray(map(wrap, outputs)))::id{NSArray}
                                          outputIndices:indices(length(outputs))::id{NSArray}
                                          weightsBuffer:nil::id{Object}
                                          perfStats:nil::id{Object}
                                          procedureIndex:keep(NSNumber(0))::id{NSNumber}]::Union{Nothing,_ANERequest})
        request === nothing && throw(ANEError("could not create an ANE request"))
        return _retain(model), _retain(request), inputs, outputs, dir
    finally
        foreach(release, temps)
    end
end

# Without `err`, a fresh error slot per call (one small allocation); for callers that
# drive a program's model and request directly.
_evaluate_native(model::id{_ANEInMemoryModel}, request::id{_ANERequest}) =
    _evaluate_native(model, request, Out{NSError}())

# `err` is reused across calls (it holds `nothing` after a successful one), so
# evaluation allocates nothing.
function _evaluate_native(model::id{_ANEInMemoryModel}, request::id{_ANERequest}, err::Out{NSError})
    @objc([model::id{_ANEInMemoryModel} evaluateWithQoS:_QOS::Cuint
                                        options:_NO_OPTIONS[]::id{NSDictionary}
                                        request:request::id{_ANERequest}
                                        error:err::Out{NSError}]::Bool) ||
        _throw_ane("ANE evaluation failed", err)
    return nothing
end

function _unload_native(model::id{_ANEInMemoryModel})
    err = Out{NSError}()
    @objc [model::id{_ANEInMemoryModel} unloadWithQoS:_QOS::Cuint error:err::Out{NSError}]::Bool
    return nothing
end

# The selectors return autoreleased (+0) objects: copy them inside a pool.
_device_string(sel) = @unsafe_autoreleasepool (p = sel(); p == nil ? nothing : String(NSString(reinterpret(id{NSString}, p))))

"""
    device_info() -> NamedTuple

Properties of this machine's Neural Engine, as reported by the ANE framework:
`has_ane`, `virtual_machine`, `num_ane`, `num_cores`, `architecture` (e.g.
`"h14g"`) and `subtype`.
"""
function device_info()
    initialize!()
    return (has_ane=@objc([_ANEDeviceInfo hasANE]::Bool),
            virtual_machine=@objc([_ANEDeviceInfo isVirtualMachine]::Bool),
            num_ane=Int(@objc [_ANEDeviceInfo numANEs]::Clong),
            num_cores=Int(@objc [_ANEDeviceInfo numANECores]::Clong),
            architecture=_device_string(() -> @objc [_ANEDeviceInfo aneArchitectureType]::id{Object}),
            subtype=_device_string(() -> @objc [_ANEDeviceInfo aneSubType]::id{Object}))
end

"""
    functional() -> Bool

Whether programs can run on a Neural Engine here: Apple Silicon, the private
framework present, and a physical (not virtualized) ANE.
"""
function functional()
    Sys.isapple() && Sys.ARCH === :aarch64 || return false
    try
        info = device_info()
        return info.has_ane && !info.virtual_machine && info.num_cores > 0
    catch
        return false
    end
end
