mutable struct AState{Q} <: TreeNodeData
    sequence :: Vector{Int}
    weights :: Vector{Float64}
    site :: Int
end

AState{Q}() where Q = AState{Q}([], [], 0)
AState() = AState{21}()

reset_weights!(x::AState) = (x.weights = [])
reset_weights!(n::TreeNode{AState}) = reset_weights!(n.data)


@kwdef mutable struct ASRMethod
    alphabet :: Symbol = :aa
    L::Int # length of sequences
    sequence_model_type :: Symbol = :profile
    sequence_model = [GTR(1., 1., 1., 1., 1., 1., .25, .25, .25, .25) for _ in 1:L]
end

