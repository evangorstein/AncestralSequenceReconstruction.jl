#######################################################################################
################################### ProfileModel{q} ###################################
#######################################################################################

"""
    ProfileModel{q} <: EvolutionModel{q}

```
P :: SVector{q, Float64}
μ :: Float64
```
An independent model without using the genetic code.
Ordering is irrelevant in this case, defaults to `1:L`.
"""
@kwdef mutable struct ProfileModel{q} <: EvolutionModel{q}
    P :: Vector{Vector{Float64}}
    μ :: Float64 = 1.
    with_code :: Bool = false
    genetic_code :: Matrix{Float64} = zeros(Float64, q, q)
    function ProfileModel{q}(P, μ, with_code, genetic_code) where q
        for p in P
            @assert isapprox(sum(p), 1) "Probabilities must sum to one - got $(sum(p))"
        end
        @assert all(p -> length(p) == q, P) "Expected probability vectors of length $q"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !with_code || q == length(AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        return new{q}(P, μ, with_code, genetic_code)
    end
end
"""
    ProfileModel(P; kwargs...)

Return an `ProfileModel` object using probability `P`. `P` can be
- a vector of probability vectors, *e.g.* `[[1/2, 1/2], [1/3, 2/3], ...]` for `q=2`
- a matrix with each columns being the probability vector for a position, *e.g.* in the same case
```
[
    1/2 1/3 ...
    1/2 2/3 ...
]
```
"""
function ProfileModel(P::AbstractVector{<:AbstractVector}; kwargs...)
    if !allequal(Iterators.map(length, P))
        error("Incorrect dimensions for probability $P")
    end

    q = length(first(P))
    return ProfileModel{q}(; P, kwargs...)
end
ProfileModel(P::AbstractMatrix; kwargs...) = ProfileModel(eachcol(P); kwargs...)

Base.length(model::ProfileModel) = length(model.P)

"""
    JukesCantor(L::Int)

The Jukes-Cantor model for sequences of length `L`.
Equivalent to `ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)`.
"""
JukesCantor(L::Int) = ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)

#=
########## ordering ##########
=#

ordering(model::ProfileModel) = 1:length(model)

#=
########## set_π ##########
=#
function set_π!(pstate::PosState{q}, model::ProfileModel{q}) where q
    for (a, x) in enumerate(model.P[pstate.pos])
        pstate.weights.π[a] = x
    end
    return nothing
end
set_π!(astate::AState, model::ProfileModel, pos::Int) = set_π!(astate.pstates[pos], model)


#=
########## set_transition_matrix ##########
=#
function set_transition_matrix_gencode!(T, model::ProfileModel{q}, t, pos) where q
    @warn "Not implemented yet!"
    return T
end

function set_transition_matrix_simple!(T, model::ProfileModel{q}, t, pos) where q
    ν = exp(-model.μ*t)
    π = model.P[pos]
    for b in 1:q
        T[:,b] .= (1-ν) * π[b]
        T[b,b] += ν
    end
    return T
end

function set_transition_matrix!(
    T::Matrix{Float64},
    model::ProfileModel{q},
    t::Number,
    pos::Int,
) where q
    return if model.with_code
        set_transition_matrix_gencode!(T, model, t, pos)
    else
        set_transition_matrix_simple!(T, model, t, pos)
    end
end
# for the root: TreeTools has branch_length(tree.root) == missing
function set_transition_matrix!(T::Matrix, model::ProfileModel, t::Missing, pos::Int)
    set_transition_matrix!(T, model, Inf, pos)
end
# π useless for this model
function set_transition_matrix!(T::Matrix, model::ProfileModel, t, pos, π)
    set_transition_matrix!(T, model, t, pos)
end

function set_transition_matrix!(astate::AState, model::EvolutionModel, t, pos::Int)
    # fallback to the form (T::Matrix, model, t, pos)
    return set_transition_matrix!(astate.pstates[pos].weights.T, model, t, pos)
end


#=
########## set_transition_rate_matrix ##########
=#

function set_transition_rate_matrix_simple!(Q, model::ProfileModel{q}, pos) where q
    π = model.P[pos]
    for a in 1:q
        Q[a, :] = π
        Q[a, a] -= 1
    end
    return Q
end
function set_transition_rate_matrix_gencode!(Q, model::ProfileModel{q}, pos) where q
    return Q
end

function set_transition_rate_matrix!(
    Q::Matrix{Float64},
    model::ProfileModel{q},
    pos::Int
) where q
    return if model.with_code
        set_transition_rate_matrix_gencode!(Q, model, pos)
    else
        set_transition_rate_matrix_simple!(Q, model, pos)
    end
end
# π useless for this model
function set_transition_rate_matrix!(Q::Matrix, model::ProfileModel, pos::Int, π)
    set_transition_rate_matrix!(Q, model, pos)
end

#=
########## Other ##########
=#

function transition_probability(old::Int, new::Int, model::ProfileModel, t, pos)
    ν = exp(-model.μ * t)
    return ν * (old == new) + (1-ν)*model.P[pos][new]
end
function transition_probability(old, new, model::ProfileModel, t, pos, π)
    return transition_probability(old, new, model, t, pos, π)
end


