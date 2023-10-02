function felsenstein!(
    root::TreeNode,
    model::EvolutionModel,
    strategy::ASRMethod,
)
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
    parent.data.pos = current_pos()
    if isleaf(parent)
        set_leaf_state!(parent.data)
        return nothing
    end
    reset_astate!(parent.data)

    # Pulling weights from all children
    for c in children(parent)
        pull_weights_up!(c, model, strategy) # pull weights for child
        verbose() > 1 && @info "Pulling weights up: from $(label(c)) to $(label(parent)) - pos $(current_pos())"
        pull_weights_up!(parent.data, c.data, branch_length(c), model, strategy)
    end

    # Special case of the root: we must set its π now since it's never called from above
    isroot(parent) && set_π!(parent.data, model)

    return nothing
end

function send_weights_down!()
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
    pull_weights_up_no_gencode!(parent::AState, child::AState, time)

Multiplies weights at `parent` by the likelihood factor coming from `child`.

## Note

Equilibrium probabilities at `child` should be set beforehand: `child.weights.π`.
"""
function pull_weights_up_no_gencode!(parent::AState{L,q}, child::AState{L,q}, time) where {L, q}
    ν = exp(-time)
    p = child.weights.π
    w = child.weights.w
    dw = (1-ν)*p'*w
    for r in 1:q
        lk_factor = ν*child.weights.w[r] + dw
        parent.weights.w[r] *= lk_factor
    end
    return nothing
end

# function pull_weights_up!(parent::AState{L,q}, child::AState{L,q})
#     lk_factor = child.weights.Q * child.weights.w
#     parent.weights.w .*= lk_factor
#     return lk_factor
# end

"""
    pull_weights_up_no_gencode_joint!(parent::AState, child::AState, time)

Equivalent to `pull_weights_up_no_gencode!` but uses the `max` instead of summing over all
states at `child`. Also sets the charactec state `child.weights.c` with the argmax.

## Note

Equilibrium probabilities at `child` should be set beforehand: `child.weights.π`.
"""
function pull_weights_up_no_gencode_joint!(parent::AState{L,q}, child::AState{L,q}, time) where {L, q}
    ν = exp(-time)
    p = child.weights.π
    w = child.weights.w
    for r in 1:q
        lk_factor, child_state = findmax(1:q) do c
            l = (1-ν)*p[c]*w[c]
            c == r && (l += ν*w[r])
            l
        end
        parent.weights.w[r] *= lk_factor
        child.weights.c[r] = child_state
    end

    return nothing
end
