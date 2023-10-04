#######################################################################################
################################## Position weights ###################################
#######################################################################################

struct PositionWeights{q}
    π :: MVector{q, Float64} # eq. probabiltiy of each state
    w :: MVector{q, Float64} # likelihood weight of each state
    P :: MMatrix{q, q, Float64} # propagator matrix to this state, given branch length to ancestor : P[a,b] = P[a-->b] = P[b|a,t]
    c :: MVector{q, Int} # character state
    function PositionWeights{q}(π, w, P, c) where q
        @assert isapprox(sum(π), 1) "Probabilities must sum to one - got $(sum(π))"
        @assert all(r -> sum(r)≈1, eachrow(P)) "Rows of transition matrix should sum to 1"
        return new{q}(π, w, P, c)
    end
end
function PositionWeights{q}(π) where q
    return PositionWeights{q}(
        π,
        ones(MVector{q, Float64}),
        MMatrix{q,q,Float64}(diagm(ones(Float64, q))),
        MVector{q}(zeros(Int, q)),
    )
end

function PositionWeights{q}() where q
    return PositionWeights{q}(
        ones(MVector{q, Float64})/q,
        ones(MVector{q, Float64}),
        MMatrix{q,q,Float64}(diagm(ones(Float64, q))),
        MVector{q}(zeros(Int, q)),
    )
end

function reset_weights!(W::PositionWeights{q}) where q
    for a in 1:q
        W.w[a] = 1
        W.c[a] = 0
        foreach(b -> W.P[a,b] = 0, 1:q)
        W.P[a,a] = 1
    end
end
function normalize!(W::PositionWeights)
    Z = sum(W.w)
    for i in eachindex(W.w)
        W.w[i] = W.w[i]/Z
    end
    return Z
end

sample(W::PositionWeights{q}) where q = StatsBase.sample(1:q, Weights(W.w))

#######################################################################################
################################### Ancestral state ###################################
#######################################################################################

@kwdef mutable struct AState{L,q} <: TreeNodeData
    # things concerning current position
    pos::Int = 1 # current position being worked on
    state::Union{Nothing, Int} = nothing
    lk::Float64 = 0. # likelihood of sampled state
    weights :: PositionWeights{q} = PositionWeights{q}()
    # things concerning the whole sequence
    sequence :: Vector{Union{Nothing, Int}} = Vector{Nothing}(undef, L) # length L
    pos_likelihood :: Vector{Float64} = Vector{Float64}(undef, L)# lk weight used at each site - length L
end

function reset_astate!(state::AState)
    state.state = nothing
    state.lk = 0
    reset_weights!(state.weights)
end

reconstructed_positions(state::AState) = findall(!isnothing, state.sequence)
is_reconstructed(state::AState, pos::Int) = !isnothing(state.sequence[pos])
hassequence(state::AState{L,q}) where {L,q} = all(i -> is_reconstructed(state, i), 1:L)


function Base.show(io::IO, ::MIME"text/plain", state::AState{L,q}) where {L,q}
    if !get(io, :compact, false)
        println(io, "Ancestral state (L: $L, q: $q)")
        println(io, "Current position: $(state.pos)")
        println(io, "Reconstructed/Observed state: $(state.state)")
        println(io, "Sequence $(state.sequence)")
    end
    return nothing
end
function Base.show(io::IO, state::AState)
    print(io, "$(typeof(state)) - \
     $(length(reconstructed_positions(state))) reconstructed positions - \
     current position $(state.pos)")
    return nothing
end

#######################################################################################
###################################### ASR Method #####################################
#######################################################################################

"""
    ASRMethod

- `joint::Bool`: joint or marginal inference. Default `false`.
- `alphabet :: Symbol`: alphabet used to map from integers to sequences. Default `:aa`.
- `verbosity :: Int`: verbosity level. Default 0.
"""
@kwdef mutable struct ASRMethod
    joint::Bool = false
    alphabet::Symbol = :aa
    verbosity::Int = 0
end
