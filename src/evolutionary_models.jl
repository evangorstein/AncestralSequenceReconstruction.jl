"""
    abstract type EvolutionModel{q}

Describes the dynamics of sequences for a state space (alphabet) of size `q`.
Concrete types of `EvolutionModel` should implement `set_π!` and `set_transition_matrix!`.
"""
abstract type EvolutionModel{q} end

# Functions below should be defined for all models

"""
    set_π!(astate::AState, model::EvolutionModel)

Set equilibrium frequencies of `astate.weights` at site `astate.pos`
using `model`.
"""
function set_π! end
"""
    set_π!(tree::Tree, model::EvolutionModel, pos::Int)

Set equilibrium frequencies at `pos` for all nodes.
"""
function set_π!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_π!(n.data.pstates[pos], model)
    end
    return nothing
end

"""
    set_transition_matrix!(T::Matrix{Float64}, model::EvolutionModel, t, pos)
    set_transition_matrix!(pstate::PState, model::EvolutionModel, t)

Set transition matrix to the input ancestral state, using branch length `t`.
In the first form, store the output in matrix `T`.
In the second form, store in  `pstate.weights.T`.
"""
function set_transition_matrix! end
"""
    set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)

Set the transition matrix for all branches in `tree` at seqeunce position `pos`.
"""
function set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_transition_matrix!(n.data, model, branch_length(n), pos)
    end
    return nothing
end

"""
    get_transition_matrix(model::EvolutionModel, t, pos)

Return the transition matrix for `model` at position `pos` and branch length `t`.
"""
function get_transition_matrix(model::EvolutionModel{q}, t, pos) where q
    T = zeros(Float64, q, q)
    set_transition_matrix!(T, model, t, pos)
    return T
end

function transition_probability end


#######################################################################################
################################### ProfileModel{q} ###################################
#######################################################################################

"""
    ProfileModel{q} <: EvolutionModel{q}

```
P :: SVector{q, Float64}
μ :: Float64
ordering :: Vector{Int}
```
An independent model without using the genetic code.
Ordering is irrelevant in this case, defaults to `1:L`.
"""
@kwdef mutable struct ProfileModel{q} <: EvolutionModel{q}
    P :: Vector{Vector{Float64}}
    μ :: Float64 = 1.
    ordering :: Vector{Int} = collect(1:length(P))
    with_code :: Bool = false
    genetic_code :: Matrix{Float64} = zeros(Float64, q, q)
    function ProfileModel{q}(P, μ, ordering, with_code, genetic_code) where q
        for p in P
            @assert isapprox(sum(p), 1) "Probabilities must sum to one - got $(sum(p))"
        end
        @assert length(ordering) == length(P) "Dimension mismatch for `P` and ordering vector"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !with_code || q == length(AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        return new{q}(P, μ, ordering, with_code, genetic_code)
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
function set_transition_matrix!(pstate::PosState, model, t::Float64)
    return set_transition_matrix!(pstate.weights.T, model, t, pstate.pos)
end
function set_transition_matrix!(astate::AState, model::EvolutionModel, t::Number, pos::Int)
    return set_transition_matrix!(astate.pstates[pos], model, t)
end

# for the root: TreeTools has branch_length(tree.root) == missing
function set_transition_matrix!(T, model::ProfileModel, t::Missing, pos::Int)
    set_transition_matrix!(T, model, Inf, pos)
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

#=
########## Other ##########
=#


"""
    transition_probability(old::Int, new::Int, model::ProfileModel, t, pos)
"""
function transition_probability(old::Int, new::Int, model::ProfileModel, t, pos)
    ν = exp(-model.μ * t)
    return ν * (old == new) + (1-ν)*model.P[pos][new]
end



