function infer_ancestral!(
    tree::Tree{<:AState},
    model::EvolutionModel,
    strategy::ASRMethod,
)

    maxL = 0 # highest achievable likelihood
    L = 0 # achieved likelihood
    for pos in model.ordering
        # Set position for all nodes in the tree
        set_pos(pos)
        foreach(n -> n.data.pos = current_pos(), nodes(tree))

        # Propagate weights up to the root
        # setting leaf state and resetting previous internal states is done here
        pull_weights_up!(tree.root, model, strategy)

        # send weights down and sample everything
        # case of the root is handled here
        send_weights_down!(tree.root, strategy)

        maxL += if strategy.joint
            maximum(tree.root.data.weights.w) |> log
        else
            sum(tree.root.data.weights.w) |> log
        end

        for node in nodes(tree)
            node.data.sequence[pos] = node.data.state
            node.data.pos_likelihood[pos] = node.data.lk
            L += log(node.data.lk)
        end
    end

    return (
        max_likelihood = maxL,
        likelihood = L,
    )
end
"""
    infer_ancestral(tree::Tree{<:AState}, model, strategy)

Create a copy of `tree` and infer ancestral states at internal nodes.
Leaves of `tree` should already be initialized with observed sequences.
"""
function infer_ancestral(tree::Tree{<:AState}, model, strategy)
    tree_copy = copy(tree)
    res = infer_ancestral!(tree_copy, model, strategy)
    return tree_copy, res
end

"""
    tree_likelihood(tree::Tree, model::EvolutionModel, strategy::ASRMethod)

Return the likelihood of the tree without performing the reconstruction.
"""
function tree_likelihood!(tree::Tree, model::EvolutionModel, strategy::ASRMethod)
    L = 0
    for pos in model.ordering
        set_pos(pos)
        foreach(n -> n.data.pos = current_pos(), nodes(tree))
        pull_weights_up!(tree.root, model, strategy)

        if strategy.joint
            L += log(maximum(
                prod,
                zip(tree.root.data.weights.π, tree.root.data.weights.w)
            ))
        else
            L += log(tree.root.data.weights.π' * tree.root.data.weights.w)
        end
    end
    return L
end

function pull_weights_up!(parent::TreeNode, model::EvolutionModel, strategy::ASRMethod)
    # This is the first time we touch `parent` for this pos: preliminary work
    verbose() > 1 && @info "Weights up for node $(label(parent)) and pos $(current_pos())"
    if isleaf(parent)
        set_leaf_state!(parent.data)
        return nothing
    end
    reset_astate!(parent.data)

    # Pulling weights from all children
    for c in children(parent)
        pull_weights_up!(c, model, strategy) # pull weights for child
        verbose() > 1 && @info "Pulling weights up: from $(label(c)) to $(label(parent)) - pos $(current_pos())"
        pull_weights_from_child!(parent.data, c.data, branch_length(c), model, strategy)
    end

    # Special case of the root: we must set its π now since it's never called from above
    if isroot(parent)
        set_π!(parent.data, model)
        set_transition_matrix!(parent.data, model, Inf)
    end

    return nothing
end



function send_weights_down!(node::TreeNode{AState{L,q}}, strategy::ASRMethod) where {L,q}
    if isroot(node)
        pull_weights_from_anc!(node.data, nothing)
        sample_node_joint!(node.data)
    else
        ancestor_state = ancestor(node).data.state
        if strategy.joint
            sample_node_joint!(node.data, ancestor_state)
        else
            pull_weights_from_anc!(node.data, ancestor_state)
            sample_node!(node.data, ancestor_state)
        end
    end

    for child in children(node)
        send_weights_down!(child, strategy)
    end

    return nothing
end

#######################################################################################
################################ Operations on weights ################################
#######################################################################################


function set_leaf_state!(leaf::AState)
    a = leaf.sequence[leaf.pos]
    for b in eachindex(leaf.weights.w)
        leaf.weights.w[b] = (b == a ? 1. : 0.)
    end
    leaf.state = a

    return nothing
end

"""
    pull_weights_from_child!(parent::AState, child::AState, t, model, strategy)

Multiply weights at `parent` by the factor coming from `child`, in Felsenstein's pruning alg
"""
function pull_weights_from_child!(
    parent::AState{L,q},
    child::AState{L,q},
    t,
    model::EvolutionModel,
    strategy::ASRMethod,
) where {L,q}
    set_π!(child, model)
    set_transition_matrix!(child, model, t)
    return if strategy.joint
        pull_weights_from_child_joint!(parent, child)
    else
        pull_weights_from_child!(parent, child)
    end
end

function pull_weights_from_child!(parent::AState{L,q}, child::AState{L,q}) where {L,q}
    lk_factor = child.weights.P * child.weights.w
    parent.weights.w .*= lk_factor
    return lk_factor
end
function pull_weights_from_child_joint!(parent::AState{L,q}, child::AState{L,q}) where {L,q}
    for r in 1:q # loop over parent state
        lk_factor, child_state = findmax(1:q) do c
            child.weights.P[r,c] * child.weights.w[c]
        end
        parent.weights.w[r] *= lk_factor
        child.weights.c[r] = child_state
    end

    return nothing
end

"""
    pull_weights_from_anc!(node::AState, ancestor_state)

Update weights of `node` using the state of its ancestor.
If `ancestor_state::Nothing`, uses the equilibrium probability distribution at `node`.
"""
pull_weights_from_anc!(node::AState, ::Nothing) = node.weights.w .*= node.weights.π
function pull_weights_from_anc!(node::AState, ancestor_state::Int)
    node.weights.w .*= node.weights.P[ancestor_state, :]
end

"""
    sample_node_joint!(node::AState, ancestor_state::Int)

Sample a state for `node` using the ancestor state. Only for the `joint` strategy.
If `ancestor_state` is not specified, use state of maximum weight (for root). 
"""
function sample_node_joint!(node::AState, ancestor_state::Int)
    node.state = node.weights.c[ancestor_state]
    node.lk = node.weights.P[ancestor_state, node.state]
    return node.state
end
function sample_node_joint!(node::AState)
    node.state = argmax(node.weights.w)
    node.lk = node.weights.π[node.state]
    return node.state
end
"""
    sample_node!(node::AState, ancestor_state)

Sample a state for node using its weights `node.weights.w`.
`ancestor_state` is used for computing the resulting probability of reconstructed state.
"""
function sample_node!(node::AState, ancestor_state::Int)
    node.state = sample(node.weights)
    node.lk = node.weights.P[ancestor_state, node.state]
    return node.state
end
function sample_node!(node::AState)
    node.state = sample(node.weights)
    node.lk = node.weights.π[node.state]
end
sample_node!(node::AState, ::Nothing) = sample_node!(node)
