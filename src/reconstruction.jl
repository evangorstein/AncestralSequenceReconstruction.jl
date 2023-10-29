function infer_ancestral!(
    tree::Tree{<:AState},
    model::EvolutionModel,
    strategy::ASRMethod,
)
    maxL = 0 # highest achievable likelihood
    likelihood = 0 # achieved likelihood
    for pos in model.ordering
        # prepare tree
        set_pos(pos)
        reset_state!(tree, pos)
        set_π!(tree, model, pos)
        set_transition_matrix!(tree, model, pos)

        # Propagate weights up to the root
        pull_weights_up!(tree.root, strategy)

        # send weights down and sample everything
        # case of the root is handled here
        send_weights_down!(tree.root, strategy)

        W = tree.root.data.pstates[pos].weights
        maxL += if strategy.joint
            log(maximum(W.v)) + W.Zv[]
        else
            log(sum(W.v)) + W.Zv[]
        end

        for node in nodes(tree)
            node.data.sequence[pos] = node.data.pstates[pos].c
            likelihood += log(node.data.pstates[pos].lk)
        end
    end

    return (
        max_likelihood = maxL,
        likelihood = likelihood,
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

function infer_ancestral(
    newick_file::AbstractString, fastafile::AbstractString, model, strategy;
    outfasta = nothing, outnewick = nothing,
)
    # read sequences
    seqmap = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    # set parameters and read tree
    L = length(first(seqmap)[2])
    q = alphabet_size(strategy.alphabet)
    if any(x -> length(x[2]) != L, seqmap)
        error("All sequences must have the same length in $fastafile")
    end
    T() = AState{q}(;L)
    tree = read_tree(newick_file; node_data_type = T)
    sequences_to_tree!(tree, seqmap)

    # re-infer branch length -- should be a strategy option
    if strategy.optimize_branch_length
        opt_strat = @set strategy.joint=false
        optimize_branch_length!(tree, model, opt_strat)
    end

    # reconstruct
    infer_ancestral!(tree, model, strategy)
    internal_sequences = map(internals(tree)) do node
        label(node) => intvec_to_sequence(node.data.sequence; alphabet = strategy.alphabet)
    end

    # write output if asked
    if !isnothing(outfasta)
        FASTAWriter(open(outfasta, "w")) do reader
            for (name, seq) in internal_sequences
                write(reader, FASTARecord(name, seq))
            end
        end
    end
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    return tree, internal_sequences
end

"""
    tree_likelihood(tree::Tree, model::EvolutionModel, strategy::ASRMethod)

Return the likelihood of the tree without performing the reconstruction.
"""
function tree_likelihood!(tree::Tree, model::EvolutionModel, strategy::ASRMethod)
    L = 0
    for pos in model.ordering
        # prepare tree
        set_pos(pos)
        reset_state!(tree, pos)
        set_transition_matrix!(tree, model, pos)

        # do the upward pass
        pull_weights_up!(tree.root, strategy)

        Wr = tree.root.data.pstates[pos].weights
        if strategy.joint
            L += log(maximum(prod, zip(Wr.π, Wr.v))) + Wr.Zv[]
        else
            L += log(Wr.π' * Wr.v) + Wr.Zv[]
        end
    end
    return L
end

function pull_weights_up!(
    parent::TreeNode{AState{q}}, strategy::ASRMethod; holder = Vector{Float64}(undef, q)
) where q
    verbose() > 2 && @info "Weights up for node $(label(parent)) and pos $(current_pos())"
    if isleaf(parent)
        set_leaf_state!(parent.data, current_pos())
        return nothing
    end

    # Pulling weights from all children
    for c in children(parent)
        pull_weights_up!(c, strategy; holder) # pull weights for child
        verbose() > 2 && @info "Pulling weights up: from $(label(c)) to $(label(parent)) - pos $(current_pos())"
        pull_weights_from_child!(
            parent.data.pstates[current_pos()],
            c.data.pstates[current_pos()],
            strategy,
            holder,
        )
    end
    normalize_weights!(parent, current_pos())

    return nothing
end



function send_weights_down!(node::TreeNode, strategy::ASRMethod)
    if isroot(node)
        pull_weights_from_anc!(node.data.pstates[current_pos()], nothing)
        if strategy.joint
            pick_node_state_joint!(node.data.pstates[current_pos()])
        else
            sample_node!(node.data.pstates[current_pos()])
        end
    else
        ancestor_state = ancestor(node).data.pstates[current_pos()].c
        if strategy.joint
            pick_node_state_joint!(node.data.pstates[current_pos()], ancestor_state)
        else
            pull_weights_from_anc!(node.data.pstates[current_pos()], ancestor_state)
            sample_node!(node.data.pstates[current_pos()], ancestor_state)
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


function set_leaf_state!(leaf::PosState, a::Int)
    for b in eachindex(leaf.weights.v)
        leaf.weights.v[b] = (b == a ? 1. : 0.)
    end
    leaf.c = a

    return nothing
end
set_leaf_state!(leaf::AState, pos) = set_leaf_state!(leaf.pstates[pos], leaf.sequence[pos])



"""
    pull_weights_from_child!(parent::PosState, child::PosState, t, model, strategy)

Multiply weights at `parent` by the factor coming from `child`, in Felsenstein's pruning alg
"""
function pull_weights_from_child!(
    parent::PosState{q},
    child::PosState{q},
    strategy::ASRMethod,
    holder::Vector{Float64} = Vector{Float64}(undef, q),
) where q
    return if strategy.joint
        pull_weights_from_child_joint!(parent, child)
    else
        pull_weights_from_child!(parent, child, holder)
    end
end

function pull_weights_from_child!(
    parent::PosState{q}, child::PosState{q}, lk_factor,
) where q
    # lk_factor = child.weights.T * child.weights.v
    mul!(lk_factor, child.weights.T, child.weights.v)
    parent.weights.v .*= lk_factor
    parent.weights.Zv[] += child.weights.Zv[]
    return lk_factor
end
function pull_weights_from_child_joint!(
    parent::PosState{q}, child::PosState{q},
) where q
    for r in 1:q # loop over parent state
        lk_factor, child_state = findmax(1:q) do c
            child.weights.T[r,c] * child.weights.v[c]
        end
        parent.weights.v[r] *= lk_factor
        child.weights.c[r] = child_state
    end
    parent.weights.Zv[] += child.weights.Zv[]

    return nothing
end

"""
    pull_weights_from_anc!(node::PosState, ancestor_state)

Update weights of `node` using the state of its ancestor.
If `ancestor_state::Nothing`, uses the equilibrium probability distribution at `node`.
"""
pull_weights_from_anc!(node::PosState, ::Nothing) = node.weights.v .*= node.weights.π
function pull_weights_from_anc!(node::PosState, ancestor_state::Int)
    node.weights.v .*= node.weights.T[ancestor_state, :]
end

"""
    pick_node_state_joint!(node::PosState, ancestor_state::Int)

Pick best state for `node` using the state of its ancestor. Only for the `joint` strategy.
If `ancestor_state` is not specified, use state of maximum weight (for root). 
"""
function pick_node_state_joint!(node::PosState, ancestor_state::Int)
    node.c = node.weights.c[ancestor_state]
    node.lk = node.weights.T[ancestor_state, node.c]
    return node.c
end
function pick_node_state_joint!(node::PosState)
    node.c = argmax(node.weights.v)
    node.lk = node.weights.π[node.c]
    return node.c
end
"""
    sample_node!(node::PosState, ancestor_state)

Sample a state for node using its weights `node.weights.v`.
`ancestor_state` is used for computing the resulting probability of reconstructed state.
"""
function sample_node!(node::PosState, ancestor_state::Int)
    node.c = sample(node.weights)
    node.lk = node.weights.T[ancestor_state, node.c]
    return node.c
end
function sample_node!(node::PosState)
    node.c = sample(node.weights)
    node.lk = node.weights.π[node.c]
end
sample_node!(node::PosState, ::Nothing) = sample_node!(node)
