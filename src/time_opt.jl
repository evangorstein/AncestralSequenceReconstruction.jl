function pupko_likelihood(node::TreeNode)
    return pupko_likelihood(node, map(s -> s.weights.T, node.data.pstates))
end
function pupko_likelihood(node::TreeNode, Ts::AbstractVector{<:AbstractMatrix{Float64}})
    return sum(zip(node.data.pstates, Ts)) do (s, T)
        log(s.weights.u' * T * s.weights.v)
    end
end

function pupko_loglk_and_grad(
    node::TreeNode,
    Qs::AbstractVector{<:AbstractMatrix{Float64}},
    Ts::AbstractVector{<:AbstractMatrix{Float64}},
)
    lks = map(zip(node.data.pstates, Ts)) do (s, T)
        s.weights.u' * T * s.weights.v
    end
    grad = sum(zip(node.data.pstates, Qs, Ts, lks)) do (s, Q, T, lk)
        (s.weights.u' * Q) * (T * s.weights.v) / lk
    end
    return sum(log, lks), grad
end

function pupko_alg!(tree::Tree, model::EvolutionModel, strategy::ASRMethod)
    for pos in model.ordering
        set_pos(pos)
        reset_state!(tree, pos)
        set_π!(tree, model, pos)
        set_transition_matrix!(tree, model, pos)

        # compute down likelihood for all nodes
        down_likelihood!(tree, strategy)

        # compute up likelihood
        up_likelihood!(tree, strategy)
    end

    return nothing
end
"""
    pupko_alg(tree::Tree, model::EvolutionModel, strategy::ASRMethod)

Apply the Pupko *et. al.* algorithm to a copy of `tree`.
For each node `n` and sequence position `pos`,
up likelihoods will be in `n.data.pstates[pos].weights.u` and the down
likelihoods in `n.data.pstates[pos].weights.v`.
"""
function pupko_alg(tree, model, strategy)
    tc = copy(tree)
    pupko_alg!(tc, model, strategy)
    return tc
end

# this is the same as in reconstruction.jl -- just an alias for clarity of the alg
down_likelihood!(tree, strategy) = pull_weights_up!(tree.root, strategy)

#
up_likelihood!(tree, strategy)  = up_likelihood!(tree.root, strategy)
function up_likelihood!(node::TreeNode{<:AState}, strategy)
    # compute up lk for `node`
    if isroot(node)
        fetch_up_lk_from_ancestor!(node.data.pstates[current_pos()], nothing)
    else
        A = ancestor(node)
        fetch_up_lk_from_ancestor!(
            node.data.pstates[current_pos()],
            A.data.pstates[current_pos()]
        )
        for c in Iterators.filter(!=(node), children(A))
            fetch_up_lk_from_child!(
                node.data.pstates[current_pos()],
                c.data.pstates[current_pos()]
            )
        end
    end
    # recursive call on children (only after we computed u)
    for c in children(node)
        up_likelihood!(c, strategy)
    end
end

function fetch_up_lk_from_ancestor!(child::PosState{q}, parent::PosState{q}) where q
    lk_factor = parent.weights.u' * parent.weights.T
    # child.weights.u .*= lk_factor
    foreach(x -> child.weights.u[x] *= lk_factor[x], 1:q)
    return lk_factor
end
function fetch_up_lk_from_ancestor!(root::PosState, ::Nothing)
    root.weights.u .*= root.weights.π
    return root.weights.π
end

function fetch_up_lk_from_child!(parent::PosState, child::PosState)
    lk_factor = child.weights.T * child.weights.v
    parent.weights.u .*= lk_factor
    return lk_factor
end
