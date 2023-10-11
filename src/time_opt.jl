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
    lw_bound = 1e-3/L
    up_bound = 2*log(L)
    epsconv = 1e-5
    maxit = 100

    opt = Opt(:LD_LBFGS, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    ftol_abs!(opt, epsconv)
    xtol_rel!(opt, epsconv)
    xtol_abs!(opt, epsconv)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)

    # Optimize
    optimize_branch_length!(node, opt, params)
end

function optimize_branch_lengths_cycle!(
    tree::Tree,
    model::EvolutionModel{q},
    strategy = ASRMethod()
) where q
    # global opt parameters
    L = length(model)
    params = (
        L = L,
        Qs = [zeros(Float64, q, q) for _ in 1:L],
        Ts = [zeros(Float64, q, q) for _ in 1:L],
        model = model,
    )
    # optimizer
    lw_bound = 1e-3/L
    up_bound = 2*log(L)
    epsconv = 1e-5
    maxit = 100

    opt = Opt(:LD_LBFGS, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    ftol_abs!(opt, epsconv)
    xtol_rel!(opt, epsconv)
    xtol_abs!(opt, epsconv)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)

    # cycle through nodes
    for n in nodes(tree; skiproot=true)
        optimize_branch_length!(n, opt, params)
        pupko_alg!(tree, model, strategy)
    end

    return nothing
end

function optimize_branch_length!(tree::Tree, model, strategy = ASRMethod(); rconv = 1e-2)
    # initial state
    pupko_alg!(tree, model, strategy)
    L = [pupko_likelihood(tree.root)]

    # first pass
    optimize_branch_lengths_cycle!(tree, model, strategy)
    push!(L, pupko_likelihood(tree.root))
    L[end] < L[end-1] && @warn "Likelihood decreased during optimization: something's wrong"

    while (L[end-1] - L[end]) / L[end-1] > rconv
        optimize_branch_lengths_cycle!(tree, model, strategy)
        push!(L, pupko_likelihood(tree.root))
        L[end] < L[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
    end

    return L
end

function optimize_branch_length!(node::TreeNode, opt, params)
    max_objective!(opt, (t, g) -> optim_wrapper(t, g, params, node))
    g = Float64[0.]
    t0 = max(Float64[branch_length(node)], opt.lower_bounds)
    result = optimize(opt, t0)
    branch_length!(node, result[2][1])
    return result
end

function optim_wrapper(t, grad, p, node)
    foreach(1:p.L) do i
        ASR.set_transition_rate_matrix!(p.Qs[i], p.model, i)
        ASR.set_transition_matrix!(p.Ts[i], p.model, t[1], i)
    end
    loglk, g = ASR.pupko_loglk_and_grad(node, p.Qs, p.Ts)
    if !isempty(grad)
        grad[1] = g
    end
    return loglk
end

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


#######################################################################################
################################### Pupko's pruning ###################################
#######################################################################################

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
