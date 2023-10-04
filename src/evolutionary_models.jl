"""
    abstract type EvolutionModel

Describes the dynamics of sequences.
Concrete types of `EvolutionModel` should implement `set_π!` and `set_transition_matrix!`.
"""
abstract type EvolutionModel end

# Functions below should be defined for all models

"""
    set_π!(astate::AState, model::EvolutionModel)

Set equilibrium frequencies of `astate.weights` at site `astate.pos`
using `model`.
"""
function set_π! end

"""
    seq_Q!(astate::AState, model::EvolutionModel, t::Float64)

Set propagator matrix `Q` to the input ancestral state, using branch length `t`.
"""
function set_transition_matrix! end

function transition_probability end


#######################################################################################
################################# ProfileModel{q} #################################
#######################################################################################

"""
    ProfileModel{q} <: EvolutionModel

```
P :: SVector{q, Float64}
μ :: Float64
ordering :: Vector{Int}
```
An independent model without using the genetic code.
Ordering is irrelevant in this case, defaults to `1:L`.
"""
@kwdef mutable struct ProfileModel{q} <: EvolutionModel
    P :: Vector{Vector{Float64}}
    μ :: Float64 = 1.
    ordering :: Vector{Int} = collect(1:length(P))
    genetic_code :: Bool = false
    function ProfileModel{q}(P, μ, ordering, genetic_code) where q
        for p in P
            @assert isapprox(sum(p), 1) "Probabilities must sum to one - got $(sum(p))"
        end
        @assert length(ordering) == length(P) "Dimension mismatch for `P` and ordering vector"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !genetic_code || q == length(AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        return new{q}(P, μ, ordering, genetic_code)
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

"""
    JukesCantor(L::Int)

The Jukes-Cantor model for sequences of length `L`.
Equivalent to `ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)`.
"""
JukesCantor(L::Int) = ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)


function set_π!(astate::AState{L,q}, model::ProfileModel{q}) where {L, q}
    for (a, x) in enumerate(model.P[astate.pos])
        astate.weights.π[a] = x
    end
    return nothing
end

"""
    set_transition_matrix!(astate, model, t)

Set transition matrix to `astate` using time `t`.
"""
function set_transition_matrix!(astate::AState{L,q}, model::ProfileModel{q}, t) where {L,q}
    ν = exp(-model.μ*t)
    π = model.P[astate.pos]
    for b in 1:q
        astate.weights.T[:,b] .= (1-ν) * π[b]
        astate.weights.T[b,b] += ν
    end
    return nothing
end

function transition_rate_matrix(astate::AState{L,q}, model::ProfileModel{q}) where {L,q}
    transition_rate_matrix(model, astate.pos)
end
function transition_rate_matrix(model::ProfileModel{q}, pos::Int) where q
    return if model.genetic_code
    else
        transition_rate_matrix_no_gencode(model, pos)
    end
end
function transition_rate_matrix_no_gencode(model::ProfileModel{q}, pos) where q
    Q = zeros(MMatrix{q,q,Float64})
    π = model.P[pos]
    for a in 1:q
        Q[a, :] = π
        Q[a, a] -= 1
    end
    return Q
end

"""
    transition_probability(old::Int, new::Int, model::ProfileModel, t, pos)
"""
function transition_probability(old::Int, new::Int, model::ProfileModel, t, pos)
    ν = exp(-model.μ * t)
    return ν * (old == new) + (1-ν)*model.P[pos][new]
end



