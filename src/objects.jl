"""
    mutable struct AState{Q} <: TreeTools.TreeNodeData

## Fields
    sequence :: Vector{Int} #
    site :: Int # current position in the sequence
    weights :: Vector{Float64} # likelihood of each state `i ∈ {1..Q}` at position `site`
    seq_weight :: Vector{Float64} # likelihood of the sampled at each sequence position
"""
mutable struct AState{Q} <: TreeNodeData
    sequence :: Vector{Int}
    site :: Int
    weights :: Vector{Float64}
    seq_weight :: Vector{Float64}
end

AState{Q}() where Q = AState{Q}([], 0, zeros(Float64, Q), Float64[])
AState() = AState{21}()

function reset_weights!(x::AState)
    for i in 1:length(x.weights)
        x.weights[i] = 1
    end
    x.site = 0
    return nothing
end
reset_weights!(n::TreeNode{<:AState}) = reset_weights!(n.data)
function reset_weights!(tree::Tree{<:AState})
    map(tree) do n
        reset_weights!(n)
        !isleaf(n) && (n.data.sequence = Int[])
    end
    return nothing
end

function normalize!(x::AState)
    x.weights /= sum(x.weights)
end

"""
    mutable struct ASRMethod

- `alphabet` - type of alphabet,  `:aa` or `:nt`. Default `:nt`.
- `L::Int` - length of sequences. Default `1`.
- `sequence_model_type` - either `:profile` or `:ardca`. Default `:profile`.
- `evolution_model` - can be an `ArNet` or a single or an array of  `SubstitutionModel`.
  Default `GTR(1., 1., 1., 1., 1., 1., .25, .25, .25, .25)` for all sites.
- `ML::Bool` : infer the maximum likelihood sequences, instead of sampling. Default `false`.
- `marginal::Bool`: infer from the likelihood marginal at each node,
  instead of taking into account the state of already sampled nodes. Default `false.`
"""
@kwdef mutable struct ASRMethod
    alphabet::Symbol = :aa
    L::Int = 1 # length of sequences
    sequence_model_type::Symbol = :profile
    evolution_model = [GTR(1., 1., 1., 1., 1., 1., .25, .25, .25, .25) for _ in 1:L]
    ML::Bool = false
    marginal::Bool = false
    normalize_weights::Bool = true

    function ASRMethod(
        alphabet, L, sequence_model_type, evolution_model::SubstitutionModel, args...,
    )
        return new(alphabet, L, sequence_model_type, [evolution_model for i in 1:L], args...)
    end
    function ASRMethod(alphabet, L, sequence_model_type, evolution_model, args...)
        return new(alphabet, L, sequence_model_type, evolution_model, args...)
    end
end
