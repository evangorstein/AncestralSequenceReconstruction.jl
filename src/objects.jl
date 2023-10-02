#######################################################################################
################################## Position weights ###################################
#######################################################################################

struct PositionWeights{q}
    π :: MVector{q, Float64} # eq. probabiltiy of each state
    w :: MVector{q, Float64} # likelihood weight of each state
    c :: MVector{q, Int} # character state
    function PositionWeights{q}(π, w, c) where q
        @assert isapprox(sum(π), 1) "Probabilities must sum to one - got $(sum(π))"
        return new{q}(π, w, c)
    end
end
function PositionWeights{q}(π) where q
    return PositionWeights{q}(π, ones(MVector{q, Float64}), MVector{q}(zeros(Int, q)))
end

function PositionWeights{q}() where q
    return PositionWeights{q}(
        ones(MVector{q, Float64})/q,
        ones(MVector{q, Float64}),
        MVector{q}(zeros(Int, q)),
    )
end

function reset_weights!(W::PositionWeights{q}) where q
    for a in 1:q
        W.w[a] = 1
        W.c[a] = 0
    end
end
function normalize!(W::PositionWeights)
    Z = sum(W.w)
    for i in eachindex(W.w)
        W.w[i] = W.w[i]/Z
    end
    return Z
end


#######################################################################################
################################### Ancestral state ###################################
#######################################################################################

@kwdef mutable struct AState{L,q} <: TreeNodeData
    # things concerning current position
    pos::Int = 1 # current position being worked on
    state::Union{Nothing, Int} = nothing
    weights :: PositionWeights{q} = PositionWeights{q}()
    # things concerning the whole sequence
    sequence :: Vector{Union{Nothing, Int}} = Vector{Nothing}(undef, L) # length L
    pos_likelihood :: Vector{Float64} = Vector{Float64}(undef, L)# lk weight used at each site - length L
end

function reset_astate!(state::AState)
    state.state = nothing
    reset_weights!(state.weights)
end

reconstructed_positions(state::AState) = findall(!isnothing, state.sequence)
is_reconstructed(state::AState, pos::Int) = !isnothing(state.sequence[pos])


function Base.show(io::IO, ::MIME"text/plain", state::AState{L,q}) where {L,q}
    if !get(io, :compact, false)
        println(io, "Ancestral state (L: $L, q: $q)")
        println(io, "Current position: $(state.pos)")
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

@kwdef mutable struct ASRMethod
    joint::Bool = false
    verbosity::Int = 0
end
