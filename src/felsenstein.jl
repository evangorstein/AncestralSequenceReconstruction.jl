#######################################################################################
####################################### Main alg ######################################
#######################################################################################

"""
    pruning_alg!(tree, model::EvolutionModel, strategy::ASRMethod)

Apply the pruning algorithm (Bousseau et. al.) to `tree` in place.
"""
function pruning_alg!(
    tree::Tree{AState{q}}, model::EvolutionModel, strategy::ASRMethod;
    set_state = true,
) where q
    if isa(model, AutoRegressiveModel) && !set_state
        error("Inconsistent `model::AutoRegressiveModel` and `set_state=false`")
    end

    holder = Vector{Float64}(undef, q) # for in place mat mul
    for pos in ordering(model)
        set_pos(pos) # set global var pos
        reset_state!(tree, pos)
        # set transition matrices for all branches
        # also sets equilibrium probabilities
        set_transition_matrix!(tree, model, pos)

        # compute down likelihood for all nodes
        down_likelihood!(tree, strategy; holder)
        # compute up likelihood
        up_likelihood!(tree, strategy; holder)

        # for each node n set n.data.pstates[pos].c :: Int, based on the strategy
        set_state && set_states!(tree, pos, strategy)
    end
    return nothing
end

"""
    pruning_alg(tree::Tree, model::EvolutionModel[, strategy::ASRMethod])

Apply the Bousseau *et. al.* algorithm to a copy of `tree`.
For each node `n` and sequence position `pos`,
up likelihoods will be in `n.data.pstates[pos].weights.u` and the down
likelihoods in `n.data.pstates[pos].weights.v`.
"""
function pruning_alg(tree, model, strategy)
    tc = copy(tree)
    pruning_alg!(tc, model, strategy)
    return tc
end

#=
## DOWN LIKELIHOOD
=#

function down_likelihood!(tree, strategy; kwargs...)
    return pull_weights_up!(tree.root, strategy; kwargs...)
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
    return if strategy.joint && strategy.ML
        pull_weights_from_child_max!(parent, child)
    else
        pull_weights_from_child_sum!(parent, child, holder)
    end
end

function pull_weights_from_child_sum!(
    parent::PosState{q}, child::PosState{q}, lk_factor,
) where q
    # lk_factor = child.weights.T * child.weights.v
    mul!(lk_factor, child.weights.T, child.weights.v)
    parent.weights.v .*= lk_factor
    parent.weights.Zv[] += child.weights.Zv[]
    return lk_factor
end
function pull_weights_from_child_max!(
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




#=
## UP LIKELIHOOD
=#


up_likelihood!(tree, strategy; kwargs...)  = up_likelihood!(tree.root, strategy; kwargs...)
function up_likelihood!(
    node::TreeNode{AState{q}}, strategy; holder = Vector{Float64}(undef, q)
) where q
    # compute up lk for `node`
    if isroot(node)
        fetch_up_lk_root!(node.data.pstates[current_pos()], strategy)
    else
        fetch_up_lk!(node, current_pos(), holder, strategy)
    end
    normalize_weights!(node, current_pos())
    # recursive call on children (only after we computed u)
    for c in children(node)
        up_likelihood!(c, strategy; holder)
    end
end

function fetch_up_lk_root!(root::PosState, strategy::ASRMethod)
    return if strategy.joint && strategy.ML
        fetch_up_lk_root_max!(root)
    else
        fetch_up_lk_root_sum!(root)
    end
end
function fetch_up_lk_root_max!(root::PosState)
    root.weights.u .= 1.
    root.weights.Zu[] = 0.
    return nothing
end
function fetch_up_lk_root_sum!(root::PosState)
    root.weights.u = root.weights.π
    root.weights.Zu[] = 0.
    return nothing
end

"""
    fetch_up_lk!(node::TreeNode, pos::Int, holder::Vector{Float64}, strategy)

Let `A` be the ancestor of `node`.
This computes the up-likelihood for the branch `A --> node`, by
- calling `fetch_up_lk_from_ancestor!(node, A)`, which will use the up lk from `A`
- calling `fetch_up_lk_from_child!(node, c)` for all `c ∈ children(A)` and `c ≠ node`,
  which will use the down lk from `c`.

If those quantities were initialized correctly, then the up likelihood at `node` is
fully computed here, but not normalized.
"""
function fetch_up_lk!(node::TreeNode, pos::Int, holder::Vector{Float64}, strategy)
    A = ancestor(node)
    fetch_up_lk_from_ancestor!(
        node.data.pstates[pos],
        A.data.pstates[pos],
        holder,
        strategy,
    )
    for c in Iterators.filter(!=(node), children(A))
        fetch_up_lk_from_child!(
            node.data.pstates[pos],
            c.data.pstates[pos],
            holder,
            strategy,
        )
    end
end



function fetch_up_lk_from_ancestor!(
    child::PosState, parent::PosState, lk_factor::Vector{Float64}, strategy::ASRMethod
)
    return if strategy.joint && strategy.ML
        fetch_up_lk_from_ancestor_max!(child, parent, lk_factor)
    else
        fetch_up_lk_from_ancestor_sum!(child, parent, lk_factor)
    end
end

function fetch_up_lk_from_ancestor_max!(
    child::PosState{q}, parent::PosState{q}, lk_factor::Vector{Float64}
) where q
    # loop over child state and find best ancestral state
    for c in 1:q
        lk_factor, a_state = findmax(1:q) do r
            parent.weights.T[r,c] * parent.weights.u[r]
        end
        child.weights.u[c] *= lk_factor
        # parent.weights.cu[c] = a_state # best state at branch above parent given c at child
    end
    child.weights.Zu[] += parent.weights.Zu[]
    return nothing
end

function fetch_up_lk_from_ancestor_sum!(
    child::PosState{q}, parent::PosState{q}, lk_factor::Vector{Float64}
) where q
    mul!(lk_factor, parent.weights.T', parent.weights.u)
    foreach(x -> child.weights.u[x] *= lk_factor[x], 1:q)
    child.weights.Zu[] += parent.weights.Zu[]
    return nothing
end

#=
in reality, parent and child are brother nodes --
I call it child because it is a child of the upper node of the branch represented by parent
ie newick `(parent,child)A`, the focus is the branch A --> parent,
and we look at contribution A --> child
=#
function fetch_up_lk_from_child!(
    parent::PosState, child::PosState, lk_factor, strategy::ASRMethod
)
    return if strategy.joint && strategy.ML
        fetch_up_lk_from_child_max!(parent, child, lk_factor)
    else
        fetch_up_lk_from_child_sum!(parent, child, lk_factor)
    end
end

function fetch_up_lk_from_child_max!(
    parent::PosState{q}, child::PosState{q}, lk_factor::Vector{Float64}
) where q
    # loop over states at parent
    for a in 1:q
        # best child state `c` given anc(parent) state `a`
        lk_factor, child_state = findmax(1:q) do c
            child.weights.T[a,c] * child.weights.v[c]
        end
        parent.weights.u[a] *= lk_factor
        # no need to set child.weights.c : already done in the down_lk pass
    end
    parent.weights.Zu[] += child.weights.Zv[]
    return nothing
end

function fetch_up_lk_from_child_sum!(
    parent::PosState, child::PosState, lk_factor::Vector{Float64}
)
    # lk_factor = child.weights.T * child.weights.v
    mul!(lk_factor, child.weights.T, child.weights.v)
    parent.weights.u .*= lk_factor
    parent.weights.Zu[] += child.weights.Zv[]
    return nothing
end




#######################################################################################
######################################## Utils ########################################
#######################################################################################


function likelihood(node::TreeNode, strategy::ASRMethod)
    return if strategy.joint && strategy.ML
        likelihood_max(node, map(s -> s.weights.T, node.data.pstates))
    else
        likelihood(node, map(s -> s.weights.T, node.data.pstates))
    end
end
function likelihood(node::TreeNode, Ts::AbstractVector{<:AbstractMatrix{Float64}})
    return sum(zip(node.data.pstates, Ts)) do (s, T)
        log(s.weights.u' * T * s.weights.v) + s.weights.Zv[] + s.weights.Zu[]
    end
end
function likelihood_max(node::TreeNode, Ts::AbstractVector{<:AbstractMatrix{Float64}})
    q = size(first(Ts), 1)
    XY = [(x,y) for x in 1:q for y in 1:q]
    return sum(zip(node.data.pstates, Ts)) do (s, T)
        lk = maximum(XY) do (x,y)
            s.weights.u[x] * s.weights.T[x,y] * s.weights.v[y]
        end
        log(lk) + s.weights.Zv[] + s.weights.Zu[]
    end
end

function set_leaf_state!(leaf::PosState, a::Int)
    for b in eachindex(leaf.weights.v)
        leaf.weights.v[b] = (b == a ? 1. : 0.)
    end
    leaf.c = a

    return nothing
end
function set_leaf_state!(leaf::PosState, ::Nothing)
    error("""Tried to initialize leaf state at position $(leaf.pos), got `nothing`.
        Are sequences attached to the leaves of the tree?"""
    )
end
set_leaf_state!(leaf::AState, pos) = set_leaf_state!(leaf.pstates[pos], leaf.sequence[pos])

function posterior(p::PosState)
    w = (p.weights.u' * p.weights.T)' .* p.weights.v
    return w / sum(w)
end
function posterior(p::PosState, anc_state::Int)
    w = p.weights.u[anc_state] * p.weights.T[anc_state,:] .* p.weights.v
    return w / sum(w)
end

"""
    pick_state_ML!(p::PosState{q}) where q

Pick marginal ML state at `p`.
"""
function pick_ML_state!(p::PosState{q}) where q
    p.posterior = posterior(p)
    p.c = argmax(p.posterior)
    return p.c
end

function pick_ML_state_joint!(p::PosState{q}) where q
    @warn "Not sure this function is working ... should take ancestral state into account"
    # error("ML + joint not implemented yet (have to fix bug) -- change strategy")
    XY = [(x,y) for x in 1:q for y in 1:q]
    lk, idx = findmax(XY) do (x,y)
        p.weights.u[x] * p.weights.T[x,y] * p.weights.v[y]
    end

    p.c = XY[idx][2]
    x = XY[idx][1]
    p.posterior = posterior(p, x)

    return p.c
end

"""
    sample_state!(pstate::PosState)

Marginally sample state, without taking ancestor state into account.
"""
function sample_state!(p::PosState)
    p.posterior = posterior(p)
    p.c = wsample(p.posterior)
    return p.c
end
sample_state!(pstate::PosState, ::Nothing) = sample_state!(pstate)

"""
    sample_state!(pstate::PosState, anc_state)

Sample state at `pstate`, taking into account sampled ancestral state.
"""
function sample_state!(p::PosState, anc_state::Int)
    p.posterior = posterior(p, anc_state)
    p.c = wsample(p.posterior)
    return p.c
end

function set_state!(pstate::PosState, anc_state::Union{Nothing, Int}, strategy::ASRMethod)
    if strategy.joint && strategy.ML
        # the joint ML reconstruction
        # this only makes sense if the alg from Pupko et. al. has been used
        pick_ML_state_joint!(pstate)
    elseif !strategy.joint && strategy.ML
        # the marginal ML reconstruction: pick max ML at p
        pick_ML_state!(pstate)
    elseif strategy.joint && !strategy.ML
        # sample at p taking the ancestor into account
        sample_state!(pstate, anc_state)
    elseif !strategy.joint && !strategy.ML
        # marginal ML: sample at p directly from the likelihood
        sample_state!(pstate)
        #
    end

    return pstate.c, pstate.posterior
end


function set_state!(node::TreeNode, anc_state, pos::Int, strategy)
    a, _ = set_state!(node.data.pstates[pos], anc_state, strategy)
    for c in children(node)
        set_state!(c, a, pos, strategy)
    end
    return nothing
end
set_states!(tree::Tree, pos::Int, strategy) = set_state!(tree.root, nothing, pos, strategy)


# useful for debugging
let
    obs_node = nothing
    global set_obs_node(n) = (obs_node = n)
    global get_obs_node() = obs_node
end
