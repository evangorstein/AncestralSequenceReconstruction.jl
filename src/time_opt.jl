function optimize_branch_length!(
    tree::Tree{<:AState}, model::ProfileModel, strategy = ASRMethod(; joint=false);
    rconv = 1e-3, ncycles = 10,
)
    set_verbose(strategy.verbosity)
    verbose() > 0 && @info "Optimizing branch length."

    # initial state
    verbose() > 1 && @info "First pass of likelihood computation..."
    t = @elapsed bousseau_alg!(tree, model, strategy)
    bousseau_alg!(tree, model, strategy)
    lk = [bousseau_likelihood(tree.root)]
    verbose() > 1 && @info "Initial lk $(lk[1]) - $t seconds"

    # first pass
    verbose() > 1 && @info "First pass of branch length opt..."
    t = @elapsed optimize_branch_lengths_cycle!(tree, model, strategy)
    push!(lk, bousseau_likelihood(tree.root))
    lk[end] < lk[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
    verbose() > 1 && @info "Likelihood $(lk) - $t seconds"

    n = 0
    while (lk[end-1] - lk[end]) / lk[end-1] > rconv && n < (ncycles - 1)
        verbose() > 1 && @info "Branch length opt $(n+2)..."
        t = @elapsed optimize_branch_lengths_cycle!(tree, model, strategy)
        push!(lk, bousseau_likelihood(tree.root))
        lk[end] < lk[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
        verbose() > 1 && @info "Likelihood $(lk) - $t seconds"
        n += 1
    end

    return lk
end
"""
    optimize_branch_length(
        tree::Tree, model::ProfileModel[, strategy::ASRMethod];
        rconv = 1e-2, ncycles = 10,
    )

Optimize branch lengths to maximize likelihood of sequences at leaves of `tree`.
"""
function optimize_branch_length(
    tree, model::ProfileModel, strategy = ASRMethod(; joint=false); kwargs...
)
    tc = copy(tree)
    lk = optimize_branch_length!(tc, model, strategy; kwargs...)
    return tc, lk
end



function optimize_branch_lengths_cycle!(
    tree::Tree,
    model::EvolutionModel{q},
    strategy = ASRMethod(; joint=false)
) where q
    # global opt parameters
    L = length(model)
    params = (
        L = L,
        Qs = [zeros(Float64, q, q) for _ in 1:L],
        Ts = [zeros(Float64, q, q) for _ in 1:L],
        model = model,
        lk_holder = Vector{Float64}(undef, L),
        qholder_1 = Vector{Float64}(undef, q),
        qholder_2 = Vector{Float64}(undef, q),
    )
    # optimizer
    lw_bound = BRANCH_LWR_BOUND(L)
    up_bound = BRANCH_UPR_BOUND(L)
    epsconv = 1e-4
    maxit = 100

    opt = Opt(:LD_LBFGS, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    # xtol_abs!(opt, epsconv)
    # xtol_rel!(opt, epsconv)
    # ftol_abs!(opt, epsconv)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)


    # cycle through nodes
    for n in Iterators.filter(!isroot, POT(tree.root))
        # set best branch length for n
        @debug "---- Opt. branch length node $(label(n)) ----"
        @debug "Previous lk" ASR.bousseau_likelihood(n)
        optimize_branch_length!(n, opt, params)

        # recompute the transition matrix for the branch above n
        foreach(1:L) do i
            ASR.set_transition_matrix!(n.data, model, branch_length(n), i)
        end
        @debug "New lk" ASR.bousseau_likelihood(n)
        bousseau_alg!(tree, model.ordering, strategy)

        @debug "Ancestor $(label(ancestor(n))) lk" ASR.bousseau_likelihood(ancestor(n))
    end

    return nothing
end

function optimize_branch_length!(node::TreeNode, opt::NLopt.Opt, params)
    max_objective!(opt, (t, g) -> optim_wrapper(t, g, params, node))
    g = Float64[0.]
    t0 = min(max(Float64[branch_length(node)], opt.lower_bounds), opt.upper_bounds)

    # =for testing=#
    # lk = optim_wrapper(t0, g, params, node)
    # @info (node=label(node), time=t0[1], lk=lk, grad=g[1])

    result = optimize(opt, t0)
    if !in(result[3], [:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED])
        @warn "Branch length opt. above $(label(node)): $result"
    end
    branch_length!(node, result[2][1])
    return result
end

function optim_wrapper(t, grad, p, node)
    foreach(1:p.L) do i
        ASR.set_transition_rate_matrix!(p.Qs[i], p.model, i) # why is this here? useless to repeat this for each time
        ASR.set_transition_matrix!(p.Ts[i], p.model, t[1], i)
    end
    loglk, g = ASR.branch_length_loglk_and_grad(
        node, p.Qs, p.Ts, p.model.μ, p.lk_holder, p.qholder_1, p.qholder_2,
    )
    if !isempty(grad)
        grad[1] = g
    end
    return loglk
end

function branch_length_loglk_and_grad(
    node::TreeNode,
    Qs::AbstractVector{<:AbstractMatrix{Float64}},
    Ts::AbstractVector{<:AbstractMatrix{Float64}},
    μ,
    lk_holder::Vector{Float64}, # dim L
    qholder_1::Vector{Float64}, # dim q
    qholder_2::Vector{Float64},
)
    for (i, (s, T)) in enumerate(zip(node.data.pstates, Ts))
        mul!(qholder_1, T, s.weights.v)
        lk_holder[i] = s.weights.u' * qholder_1
    end
    loglk = sum(log, lk_holder)
    if isnan(loglk)
        @info lk_holder, map(s -> (s.weights.u', s.weights.v), node.data.pstates)
        error("Error when computing likelihood: encountered `NaN` - node $(label(node))")
    end

    grad = sum(zip(node.data.pstates, Qs, Ts, lk_holder)) do (s, Q, T, lk)
        mul!(qholder_1, T, s.weights.v)
        mul!(qholder_2, Q, qholder_1)
        μ * s.weights.u' * qholder_2 / lk
    end

    return loglk, grad
end

function bousseau_likelihood(node::TreeNode)
    return bousseau_likelihood(node, map(s -> s.weights.T, node.data.pstates))
end
function bousseau_likelihood(node::TreeNode, Ts::AbstractVector{<:AbstractMatrix{Float64}})
    return sum(zip(node.data.pstates, Ts)) do (s, T)
        log(s.weights.u' * T * s.weights.v) + s.weights.Zv[] + s.weights.Zu[]
    end
end


"""
    update_neighbours!(node::TreeNode)

    WONT WORK -- keeping for now but should not be used
"""
function update_neighbours!(node::TreeNode; kwargs...)
    return foreach(i -> update_neighbours!(node, i; kwargs...), 1:node.data.L)
end
function update_neighbours!(
    node::TreeNode, pos::Int;
    sisters = false, anc = false, child = false,
)
    @warn "update_neighbours! should probably not be used"
    @assert !isroot(node) "This should never be called on the root node"
    if sisters
        # sisters of node: likelihood up `u` has to be recomputed
        # node will be involved as: node.T * node.v
        for sister in Iterators.filter(!=(node), children(ancestor(node)))
            reset_up_likelihood!(sister, pos)
            fetch_up_lk!(sister, pos)
            normalize_weights!(sister, pos)
        end
    end
    if anc
        # ancestor of node: likelihood down `v` has to be recomputed
        # node will be involved as: node.T * node.v
        A = ancestor(node)
        reset_down_likelihood!(A, pos)
        for c in children(A)
            pull_weights_from_child!(A.data.pstates[pos], c.data.pstates[pos])
        end
        normalize_weights!(A, pos)
    end

    if child
        # children of node: likelihood up `u` has to be recomputed
        # node will be involved as: node.T * node.u
        for c in children(node)
            reset_up_likelihood!(c, pos)
            fetch_up_lk!(c, pos)
            normalize_weights!(c, pos)
        end
    end
end

#=
Optimize a single branch length (no cycle)
useful only for testing
check code before running, not up to date with the other optimize_branch_length!
=#
function optimize_branch_length!(node::TreeNode, model::EvolutionModel{q}) where q
    L = length(node.data.pstates)
    # Set parameters
    params = (
        L = L,
        Qs = [zeros(Float64, q, q) for _ in 1:L],
        Ts = [zeros(Float64, q, q) for _ in 1:L],
        model = model,
    )

    # Set optimizer
    # optimizer
    lw_bound = BRANCH_LWR_BOUND(L)
    up_bound = BRANCH_UPR_BOUND(L)
    epsconv = 1e-4
    maxit = 100

    opt = Opt(:LD_LBFGS, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    # xtol_abs!(opt, epsconv)
    # xtol_rel!(opt, epsconv)
    # ftol_abs!(opt, epsconv)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)

    # Optimize
    @debug "---- Opt. branch length node $(label(node)) ----"
    @debug "Previous lk" ASR.bousseau_likelihood(node)

    optimize_branch_length!(node, opt, params)


    # recompute the transition matrix for the branch above n
    foreach(1:L) do i
        ASR.set_transition_matrix!(node.data, model, branch_length(node), i)
    end
    @debug "New lk" ASR.bousseau_likelihood(node)
    # update_neighbours!(node; anc=true, sisters=true)
    # bousseau_alg!


    @debug "Ancestor $(label(ancestor(node))) lk" ASR.bousseau_likelihood(ancestor(node))
    for c in children(ancestor(node))
        @debug "sister $(label(c)) lk" ASR.bousseau_likelihood(c)
    end
end


#######################################################################################
################################### Bousseau's pruning ###################################
#######################################################################################

"""
    bousseau_alg!(tree, model::EvolutionModel, strategy::ASRMethod)
    bousseau_alg!(tree, ordering, strategy::ASRMethod)

First form: apply the Bousseau et. al. algorithm to `tree` in place.
Second form: same, but assumes that the transition matrices and equilibrium frequencies
for each node are **already set**.
"""
function bousseau_alg!(
    tree::Tree, model::EvolutionModel, strategy::ASRMethod = ASRMethod(; joint=false)
)
    if strategy.joint
        error("For now, `bousseau_alg` is only for marginal: `strategy.joint=false`")
    end

    for pos in model.ordering
        set_π!(tree, model, pos)
        set_transition_matrix!(tree, model, pos)
    end
    bousseau_alg!(tree, model.ordering, strategy)

    return nothing
end
function bousseau_alg!(
    tree::Tree{AState{q}}, ordering::AbstractVector{Int}, strategy::ASRMethod,
) where q
    holder = Vector{Float64}(undef, q) # for in place mat mul
    for pos in ordering
        set_pos(pos)
        reset_state!(tree, pos)

        # compute down likelihood for all nodes
        down_likelihood!(tree, strategy; holder)
        # compute up likelihood
        up_likelihood!(tree, strategy; holder)
    end
    return nothing
end

"""
    bousseau_alg(tree::Tree, model::EvolutionModel[, strategy::ASRMethod])

Apply the Bousseau *et. al.* algorithm to a copy of `tree`.
For each node `n` and sequence position `pos`,
up likelihoods will be in `n.data.pstates[pos].weights.u` and the down
likelihoods in `n.data.pstates[pos].weights.v`.
"""
function bousseau_alg(tree, model, strategy = ASRMethod(; joint=false))
    tc = copy(tree)
    bousseau_alg!(tc, model, strategy)
    return tc
end

# this is the same as in reconstruction.jl -- just an alias for clarity of the alg
function down_likelihood!(tree, strategy; kwargs...)
    return pull_weights_up!(tree.root, strategy; kwargs...)
end

#
up_likelihood!(tree, strategy; kwargs...)  = up_likelihood!(tree.root, strategy; kwargs...)
function up_likelihood!(
    node::TreeNode{AState{q}}, strategy; holder = Vector{Float64}(undef, q)
) where q
    # compute up lk for `node`
    if isroot(node)
        fetch_up_lk_from_ancestor!(node.data.pstates[current_pos()], nothing)
    else
        fetch_up_lk!(node, current_pos(), holder)
        normalize_weights!(node, current_pos())
    end
    # recursive call on children (only after we computed u)
    for c in children(node)
        up_likelihood!(c, strategy; holder)
    end
end

"""
    fetch_up_lk!(node::TreeNode, pos::Int)

Let `A` be the ancestor of `node`.
This computes the up-likelihood for the branch `A --> node`, by
- calling `fetch_up_lk_from_ancestor!(node, A)`, which will use the up lk from `A`
- calling `fetch_up_lk_from_child!(node, c)` for all `c ∈ children(A)` and `c ≢ node`,
  which will use the down lk from `c`.

If those quantities were initialized correctly, then the up likelihood at `node` is
fully computed here, but not normalized.
"""
function fetch_up_lk!(node::TreeNode, pos::Int, holder::Vector{Float64})
    A = ancestor(node)
    fetch_up_lk_from_ancestor!(
        node.data.pstates[pos],
        A.data.pstates[pos],
        holder,
    )
    for c in Iterators.filter(!=(node), children(A))
        fetch_up_lk_from_child!(
            node.data.pstates[pos],
            c.data.pstates[pos],
            holder,
        )
    end
end

function fetch_up_lk_from_ancestor!(
    child::PosState{q}, parent::PosState{q}, lk_factor::Vector{Float64}
) where q
    # lk_factor = parent.weights.u' * parent.weights.T
    mul!(lk_factor, parent.weights.T', parent.weights.u)
    foreach(x -> child.weights.u[x] *= lk_factor[x], 1:q)
    child.weights.Zu[] += parent.weights.Zu[]
    return lk_factor
end
function fetch_up_lk_from_ancestor!(root::PosState, ::Nothing)
    root.weights.u .*= root.weights.π
    return root.weights.π
end

#=
in reality, parent and child are brother nodes --
I call it child because it is a child of the upper node of the branch represented by parent
ie `(parent,child)A`, and the focus is the branch A --> parent
=#
function fetch_up_lk_from_child!(
    parent::PosState, child::PosState, lk_factor::Vector{Float64}
)
    # lk_factor = child.weights.T * child.weights.v
    mul!(lk_factor, child.weights.T, child.weights.v)
    parent.weights.u .*= lk_factor
    parent.weights.Zu[] += child.weights.Zv[]
    return lk_factor
end

