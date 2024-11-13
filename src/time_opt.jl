###########################################################################################
####################################### BRANCH OPT ########################################
###########################################################################################


"""
    optimize_branch_length(
        newick_file::AbstractString,
        fastafile::AbstractString,
        model::EvolutionModel,
        strategy = ASRMethod(; joint=false);
        outnewick=nothing,
    )
"""
function optimize_branch_length(
    newick_file::AbstractString,
    fastafile::AbstractString,
    model::EvolutionModel{q},
    strategy = ASRMethod(; joint=false);
    outnewick=nothing,
    kwargs...
) where q
    # read sequences
    seqmap = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    # set parameters and read tree
    L = length(first(seqmap)[2])
    if any(x -> length(x[2]) != L, seqmap)
        error("All sequences must have the same length in $fastafile")
    end
    T() = AState{q}(;L)
    tree = read_tree(newick_file; node_data_type = T)
    sequences_to_tree!(tree, seqmap; alphabet=model.alphabet)

    # re-infer branch lengths
    opt_strat = @set strategy.joint=false
    optimize_branch_length!(tree, model, opt_strat; kwargs...)

    # write output if needed
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    return tree
end

function optimize_branch_length!(tree::Tree, model::AutoRegressiveModel, strat; kwargs...)
    @warn """
        Cannot optimize branches of tree using an autoregressive model.
        Constructing a profile model using the autoregressive, and using it for branch
        length optimization.
        """
    profile_model = ProfileModel(model)
    return optimize_branch_length!(tree, profile_model, strat; kwargs...)
end

function optimize_branch_length!(
    tree::Tree{AState{q}}, model::ProfileModel{q}, strategy = ASRMethod(; joint=false);
    rconv = 1e-3,
) where q
    set_verbose(strategy.verbosity)
    verbose() > 0 && @info "Optimizing branch length."
    verbose() > 2 && @info "Branch lengths" map(branch_length, tree)

    # initial state
    verbose() > 1 && @info "First pass of likelihood computation..."
    t = @elapsed pruning_alg!(tree, model, strategy; set_state=false)
    lk = [likelihood(tree.root, strategy)]
    verbose() > 1 && @info "Initial lk $(lk[1]) - $t seconds"

    # first pass
    verbose() > 1 && @info "First pass of branch length opt..."
    t = @elapsed optimize_branch_lengths_cycle!(tree, model, strategy)
    push!(lk, likelihood(tree.root, strategy))
    lk[end] < lk[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
    rel_delta_lk = (lk[end-1] - lk[end]) / lk[end-1]
    verbose() > 1 && @info "Likelihood $(lk) - $t seconds"
    verbose() > 1 && @info "Relative lk increase: $(rel_delta_lk)"
    verbose() > 2 && @info "Branch lengths" map(branch_length, tree)

    n = 0
    while abs(rel_delta_lk) > rconv && n < (strategy.optimize_branch_length_cycles - 1)
        verbose() > 1 && @info "Branch length opt $(n+2)..."
        t = @elapsed optimize_branch_lengths_cycle!(tree, model, strategy)
        push!(lk, likelihood(tree.root, strategy))
        lk[end] < lk[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
        rel_delta_lk = (lk[end-1] - lk[end]) / lk[end-1]
        verbose() > 1 && @info "Likelihood $(lk) - $t seconds"
        verbose() > 1 && @info "Relative lk increase: $(rel_delta_lk)"
        verbose() > 2 && @info "Branch lengths" map(branch_length, tree)
        n += 1
    end

    return lk
end
"""
    optimize_branch_length(
        tree::Tree, model::ProfileModel[, strategy::ASRMethod];
        rconv = 1e-2
    )

Optimize branch lengths to maximize likelihood of sequences at leaves of `tree`.
"""
function optimize_branch_length(
    tree, model::EvolutionModel, strategy = ASRMethod(; joint=false); kwargs...
)
    tc = copy(tree)
    lk = optimize_branch_length!(tc, model, strategy; kwargs...)
    return tc, lk
end



function optimize_branch_lengths_cycle!(
    tree::Tree,
    model::ProfileModel{q},
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
    lw_bound = BRANCH_LWR_BOUND(L; style=:ml)
    up_bound = BRANCH_UPR_BOUND(model; style=:bayes)
    epsconv = 1e-2
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
    for n in Iterators.filter(!isroot, POT(tree))
        # set best branch length for n
        @debug "\n---- Opt. branch length node $(label(n)) ----"
        @debug "Previous lk" ASR.likelihood(n, strategy)
        @debug "Previous branch length $(branch_length(n))"
        optimize_branch_length!(n, opt, params)

        # recompute the transition matrix for the branch above n
        foreach(1:L) do i
            ASR.set_transition_matrix!(
                n.data, model, branch_length(n), i; set_equilibrium_frequencies=false
            )
        end
        pruning_alg!(tree, model, strategy; set_state=false)
        @debug "New lk" ASR.likelihood(n, strategy)
        @debug "New branch length $(branch_length(n))"
        @debug "Branch length bounds $(lw_bound) < $(up_bound)"

        @debug "Ancestor $(label(ancestor(n))) lk" ASR.likelihood(ancestor(n), strategy)
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

###########################################################################################
##################################### BRANCH SCALING ######################################
###########################################################################################
function optimize_branch_scale(
    newick_file::AbstractString,
    fastafile::AbstractString,
    model::EvolutionModel{q},
    strategy = ASRMethod(; joint=false);
    outnewick=nothing,
) where q
    # read sequences
    seqmap = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    # set parameters and read tree
    L = length(first(seqmap)[2])
    if any(x -> length(x[2]) != L, seqmap)
        error("All sequences must have the same length in $fastafile")
    end
    T() = AState{q}(;L)
    tree = read_tree(newick_file; node_data_type = T)
    sequences_to_tree!(tree, seqmap; alphabet=model.alphabet)

    # re-infer branch lengths
    optimize_branch_scale!(tree, model, strategy)

    # write output if needed
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    return tree
end

function optimize_branch_scale!(tree::Tree, model::EvolutionModel, strategy)
    set_verbose(strategy.verbosity)

    # parameters
    params = (tree = tree, model = model, strategy = strategy)

    # optimizer
    lw_bound = 1e-5
    up_bound = 1e5
    epsconv = 1e-5
    maxit = 200

    # opt = Opt(:LN_COBYLA, 1)
    opt = Opt(:LN_SBPLX, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)

    max_objective!(opt, (μ, g) -> optim_wrapper_branch_scale!(μ, g, params))
    μ0 = Float64[1]

    verbose() > 0 && @info """
        Finding optimal scaling for branch length using model of type $(typeof(model))
        """
    verbose() > 1 && @info "Initial tree likelihood $(tree_likelihood!(tree, model, params.strategy))"
    result = optimize(opt, μ0)
    if !in(result[3], [:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED])
        @warn "Branch scaling: $result"
    end
    verbose() > 1 && @info "Found scaling $(result[2][1])"
    verbose() > 1 && @info "Final log likelihood $(result[1])"

    scale_branches!(tree, result[2][1])
    return result
end

function optim_wrapper_branch_scale!(μ, grad, params)
    scale_branches!(params.tree, μ[1])
    loglk = tree_likelihood!(params.tree, params.model, params.strategy)
    scale_branches!(params.tree, 1/μ[1])
    # I think I always need this
    if length(grad)>0
    end
    return loglk
end

function scale_branches!(tree::Tree, μ::Number)
    foreach(n -> branch_length!(n, μ*branch_length(n)), nodes(tree; skiproot=true))
end

###########################################################################################
################################# OLD CODE / FOR TESTING ##################################
###########################################################################################

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
    @debug "Previous lk" ASR.likelihood(node)

    optimize_branch_length!(node, opt, params)


    # recompute the transition matrix for the branch above n
    foreach(1:L) do i
        ASR.set_transition_matrix!(node.data, model, branch_length(node), i)
    end
    @debug "New lk" ASR.likelihood(node)



    @debug "Ancestor $(label(ancestor(node))) lk" ASR.likelihood(ancestor(node))
    for c in children(ancestor(node))
        @debug "sister $(label(c)) lk" ASR.likelihood(c)
    end
end




#========================#
######### Bounds #########
#========================#

# using the idea (1 - e(-t)) ∼ (n+1) / (L+2) (fraction of mutated sites, with a pc.)
# for n = L and n = 0 we get the limits below
BRANCH_LWR_BOUND_BAYES(L) = log(L+2) - log(L+1)
BRANCH_UPR_BOUND_BAYES(L) = log(L+2) * 0.75 # this over-estimates since saturation occurs before -- would need to put long term eq. of model

function BRANCH_UPR_BOUND_BAYES(profile::ProfileModel)
    L = length(profile)
    H_inf = sum(x -> x^2, Iterators.flatten(profile.P))
    return log(H_inf+2)*0.75
end

BRANCH_LWR_BOUND_ML(L) = 0
BRANCH_UPR_BOUND_ML(L) = Inf

function BRANCH_LWR_BOUND(L; style = :ML)
    return if style == :bayes
        BRANCH_LWR_BOUND_BAYES(L)
    elseif style == :ml || style == :ML
        BRANCH_LWR_BOUND_ML(L)
    else
        error("Unknown style $style")
    end
end
function BRANCH_UPR_BOUND(L; style = :bayes)
    return if style == :bayes
        BRANCH_UPR_BOUND_BAYES(L)
    elseif style == :ml || style == :ML
        BRANCH_UPR_BOUND_ML(L)
    else
        error("Unknown style $style")
    end
end
