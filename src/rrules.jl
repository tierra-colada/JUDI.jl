############################ AD rules ############################################
function rrule(F::judiPropagator{T, O}, m::AbstractArray{T}, q::AbstractArray{T}) where {T, O}
    y = F(m, q)
    dims = length(q)
    function pullback(Δy)
        dq = @thunk(diffsource(F, m, Δy, dims))
        dm = @thunk(reshape(judiJacobian(F, q)'(m, Δy), size(m)))
        return (NoTangent(), dm, dq)
    end
    return y, pullback
end

function diffsource(F::judiPropagator{T, O}, m::AbstractArray{T}, Δy, dims::Integer) where {T, O}
    ra = F.options.return_array
    dq = F'(m, Δy)
    # Reshape if vector
    dq = ra ? reshape(dq, dims) : dq
    return dq
end


# projection
(project::ProjectTo{AbstractArray})(dx::PhysicalParameter) = project(reshape(dx.data, project.axes))

############################# mul ##########################################
# Array with additional channel and batch dim
*(F::judiPropagator, q::Array{T, 4}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 5}) where T = F*vec(q)
*(F::judiPropagator, q::Array{T, 6}) where T = F*vec(q)
