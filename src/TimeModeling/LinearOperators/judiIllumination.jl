
"""
    judiIllumination
Illumination compensation operator. Can only be used in combination with a judiJacobian if uninitialized.

This operator will store the illumination correction for the latest application of J'.


"""
struct judiIllumination{DDT} <: joAbstractLinearOperator{DDT, DDT}
    name::String
    illum::Vector{DDT}
    m::Integer
    n::Integer
    fop::Function
    fop_T::Function
end

conj(I::judiIllumination{T}) where T = judiIllumination{T}(I.name, conj(I.illum), I.fop, I.fop_T)
adjoint(I::judiIllumination{T}) where T = judiIllumination{T}(I.name, conj(I.illum), I.fop, I.fop_T)
transpose(I::judiIllumination{T}) where T = judiIllumination{T}(I.name, I.illum, I.fop, I.fop_T)


function apply_diag(I::judiIllumination, x)
    out = deepcopy(x)
    out[I.illum > eps(T)] ./= I.illum[I.illum > eps(T)]
    out
end

function judiIllumination(name::String, model::Model) where T
    n = prod(model.n)
    illum = Vector{T}(undef, n)
    judiIllumination{T}(name, illum, n, n, x->apply_diag(J, x), x->apply_diag(J, x))
end

judiIllumination(model::Model) = judiIllumination(name="Illumination correction", model)

for JT ∈ [judiModeling, judiAbstractJacobian, judiPDEfull, judiPDEextended, judiPDE]
    @eval judiIllumination(F::$(JT)) = judiIllumination(F.model.n)
end

