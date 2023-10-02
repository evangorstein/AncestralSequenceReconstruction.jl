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

"""
    pull_weights_up!(parent::AState, child::AState, t, model, strategy)

Multiply weights at `parent` by the factor coming from `child`, in Felsenstein's pruning alg
"""
function pull_weights_up! end

#######################################################################################
################################# IndependentModel{q} #################################
#######################################################################################

"""
    IndependentModel{q} <: EvolutionModel

```
P :: SVector{q, Float64}
μ :: Float64
ordering :: Vector{Int}
```
An independent model without using the genetic code.
Ordering is irrelevant in this case, defaults to `1:L`.
"""
@kwdef mutable struct IndependentModel{q} <: EvolutionModel
    P :: Vector{SVector{q, Float64}}
    μ :: Float64 = 1.
    ordering :: Vector{Int} = collect(1:length(P))
    genetic_code :: Bool = false
    function IndependentModel{q}(P, μ, ordering, genetic_code) where q
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
    IndependentModel(P; kwargs...)

Return an `IndependentModel` object using probability `P`. `P` can be
- a vector of probability vectors, *e.g.* `[[1/2, 1/2], [1/3, 2/3], ...]` for `q=2`
- a matrix with each columns being the probability vector for a position, *e.g.* in the same case
```
[
    1/2 1/3 ...
    1/2 2/3 ...
]
```
"""
function IndependentModel(P::AbstractVector{<:AbstractVector}; kwargs...)
    if !allequal(Iterators.map(length, P))
        error("Incorrect dimensions for probability $P")
    end

    q = length(first(P))
    return IndependentModel{q}(; P, kwargs...)
end
IndependentModel(P::AbstractMatrix; kwargs...) = IndependentModel(eachcol(P); kwargs...)

"""
    JukesCantor(L::Int)

The Jukes-Cantor model for sequences of length `L`.
Equivalent to `IndependentModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)`.
"""
JukesCantor(L::Int) = IndependentModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)


function set_π!(astate::AState{L,q}, model::IndependentModel{q}) where {L, q}
    for (a, x) in enumerate(model.P[astate.pos])
        astate.weights.π[a] = x
    end
    return nothing
end

"""
    set_transition_matrix!(astate, model, t)

Set transition matrix to `astate` using time `t`.
"""
function set_transition_matrix!(astate::AState{L,q}, model::IndependentModel{q}, t) where {L,q}
    ν = exp(-model.μ*t)
    π = model.P[astate.pos]
    for a in 1:q
        astate.weights.P[a,:] .= (1-ν) * π
        astate.weights.P[a,a] += ν
    end
    return nothing
end




function pull_weights_up!(
    parent::AState{L,q},
    child::AState{L,q},
    t,
    model::IndependentModel{q},
    strategy::ASRMethod,
) where {L,q}
    set_π!(child, model)
    set_transition_matrix!(child, model, t)
    return if strategy.joint
        pull_weights_up_joint!(parent, child)
    else
        pull_weights_up!(parent, child)
    end
end

"""
    transition_probability(old::Int, new::Int, model::IndependentModel, t, pos)
"""
function transition_probability(old::Int, new::Int, model::IndependentModel, t, pos)
    ν = exp(-model.μ * t)
    return ν * (old == new) + (1-ν)*model.P[pos][new]
end



