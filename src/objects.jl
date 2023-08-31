mutable struct AState{Q} <: TreeNodeData
    sequence :: Vector{Int}
    weights :: Vector{Float64}
    p :: Float64
    site :: Int
end

AState{Q}() where Q = AState{Q}([], zeros(Float64, Q), 0., 0)
AState() = AState{21}()

function reset_weights!(x::AState)
    for i in 1:length(x.weights)
        x.weights[i] = 1
    end
    x.p = 0.
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

@kwdef mutable struct ASRMethod
    alphabet::Symbol = :aa
    L::Int = 1 # length of sequences
    sequence_model_type::Symbol = :profile
    sequence_model = [GTR(1., 1., 1., 1., 1., 1., .25, .25, .25, .25) for _ in 1:L]
    ML::Bool = false
    marginal::Bool = false
    normalize_weights::Bool = true

    function ASRMethod(
        alphabet, L, sequence_model_type, sequence_model::SubstitutionModel, args...,
    )
        return new(alphabet, L, sequence_model_type, [sequence_model for i in 1:L], args...)
    end
    function ASRMethod(alphabet, L, sequence_model_type, sequence_model, args...)
        return new(alphabet, L, sequence_model_type, sequence_model, args...)
    end
end
