#=
In SubstitutionModel, P(model, Δt) is the matrix
Pab = P[b | a, Δt]

A sum like Σ_b P(b | a, t) * L(b) is `P*L`
=#

function infer_ancestral!(tree::Tree{AState}, method::ASRMethod)
    for site in 1:L
        fetch_weights_down!(tree.root, site, method)
        pull_weights_sample!(tree.root, site, method)
    end
    return tree
end
infer_ancestral(t::Tree{AState}, method::ASRMethod) = infer_ancestral!(copy(t), method)
function infer_ancestral(t::Tree, method::ASRMethod)
    Q = method.alphabet in aa_alphabet_names ? length(AA_ALPHABET) : length(NT_ALPHABET)
    tc = convert(Tree{AState{Q}}, t)
    return infer_ancestral!(tc, method)
end

"""
    fetch_weights_down!(n::TreeNode, site::Int, method::ASRMethod)

Get likelihood weights from the children of `n`.
Calls recursively down the tree.
If `n` is a leaf, get its weights from its sequence.
"""
function fetch_weights_down!(n::TreeNode{<:AState}, site::Int, method::ASRMethod)
    if n.data.site == site
        throw(ErrorException("$(n.label) already initialized?"))
    end
    reset_weights!(n)
    n.data.site = site

    if isleaf(n)
        if length(n.data.sequence) < site
            throw(ErrorException("""Issue with sequence on leaf $n at pos $site:\
                sequence has length $(length(n.data.sequence))
            """))
        end
        set_weights_from_sequence!(n.data, site)
    else
        for c in children(n)
            fetch_weights_down!(c, site, method)
            fetch_weights!(n, c, site, method)
            method.normalize_weights && normalize!(n.data)
        end
    end

    return n.data
end

"""
    pull_weights_sample!(n::TreeNode, site, method)

- pull weights from `ancestor(n)` (if not a leaf)
- sample state at position `site` for `n`
- call self on `children(n)`
"""
function pull_weights_sample!(n::TreeNode{<:AState}, site, method)
    isleaf(n) && return nothing

    # Computing weights for `n` using `ancestor(n)`
    if !isroot(n)
        fetch_weights_up!(n, site, method)
        method.normalize_weights && normalize!(n.data)
    end

    # sample sequence at site
    if length(n.data.sequence) != site -1
        throw(ErrorException("Sequence of $n already sampled at site $site - $(n.data)"))
    end
    a, w = if method.ML
        findmax(n.data.weights)[2]
    else
        ws = Weights(n.data.weights)
        a = sample(ws)
        a, ws[a]
    end
    push!(n.data.sequence, a)
    n.data.p = w

    # potentially collapse weights
    # if method is marginal, we don't: ignoring reconstruction at other sites
    if !method.marginal
        set_weights_from_sequence!(n.data, site)
    end

    for c in children(n)
        pull_weights_sample!(c, site, method)
    end

    return nothing
end

"""
    fetch_weights_up(n::TreeNode{<:AState}, site::Int, method::ASRMethod)

Given that `ancestor(n)` already has an inferred state `a` at `site`, pick the state of `n`
by sampling from
```
Qt[b | a] * n.data.weights[b]
```
"""
function fetch_weights_up!(n::TreeNode{<:AState}, site::Int, method::ASRMethod)
    if isroot(n)
        throw(ErrorException("Cannot fetch weights up for root node $(n.label)"))
    end

    An = ancestor(n)
    # An should already have something inferred at this position
    if length(An.data.sequence) != site
        throw(ErrorException("""Ancestor of $n has an unexpected sequence length.\
            Sequence of length $(length(An.data.sequence)) but reconstructing site $site.
        """))
    end

    Δt = branch_length(n)
    if method.sequence_model_type == :profile
        fetch_weights_up_profile!(n.data, An.data, Δt, site, method)
    elseif method.sequence_model_type == :ArDCA
        fetch_weights_up_ardca!()
    else
        throw(ArgumentError("Unrecognized sequence model $(method.sequence_model_type)"))
    end
end

function fetch_weights_up_profile!(dest, source, Δt, site, method)
    model = method.sequence_model[site]
    Qt = P(model, Δt)' # we multiply from left
    dw = Qt * source.weights
    dest.weights .*= dw
    return dw
end

"""
    fetch_weights!(dest::TreeNode, source::TreeNode, site, method::ASRMethod)

Compute contribution of likelihood weights of `source` to `dest`.

If `method.ML`, then this is
```
max_b Qt[a, b] * source.data.weights[b]
```

If `!method.ML`, then
```
sum_b Qt[a, b] * source.data.weights[b]
```
"""
function fetch_weights!(
    dest::TreeNode{<:AState}, source::TreeNode{<:AState}, site::Int, method::ASRMethod
)
    if source.data.site != site
        throw(ErrorException("""Cannot fetch weights from $(source.label) to $(dest.label):\
             the latter is uninitialized.
        """))
    end

    # Getting branch length
    Δt = if ancestor(source) == dest
        branch_length(source)
    elseif !isroot(dest) && ancestor(dest) == source
        branch_length(dest)
    else
        throw(ErrorException("""Cannot fetch weights from $(source.label) to $(dest.label):\
             the nodes are not connected by a branch.
        """))
    end

    # Calling specific functions for different evolution models
    if method.sequence_model_type == :profile
        fetch_weights_profile!(dest.data, source.data, Δt, site, method)
    elseif method.sequence_model_type == :ArDCA
        fetch_weights_ardca!()
    else
        throw(ArgumentError("Unrecognized sequence model $(method.sequence_model_type)"))
    end
end
function fetch_weights_profile!(dest, source, Δt, site, method)
    model = method.sequence_model[site]
    Qt = P(model, Δt)
    Δw = if method.ML
        dw = similar(source.weights)
        for a in 1:length(dw)
            dw[a] = findmax(b -> Qt[a, b] * source.weights[b], 1:length(dw))[1]
        end
        dw
    else
        Qt * source.weights
    end
    dest.weights .*= Δw
    return Δw
end

function set_weights_from_sequence!(x::AState, site::Int)
    for i in 1:length(x.weights)
        x.weights[i] = 0
    end
    x.weights[x.sequence[x.site]] = 1
    return nothing
end
