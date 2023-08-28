function fetch_weights_down(n::TreeNode{AState{Q}}, site::Int, method::ASRMethod) where Q
    if isleaf(n)
        # leaves should have a sequence
        isempty(n.data.sequence) && throw(ErrorException("Leaf $(n.label) has no sequence"))
        if length(n.data.sequence) < site
            throw(ErrorException("""
            Can't reconstruct site $site for sequences of length $(length(n.data.sequence))
            """))
        end
        #
        n.data.site = site
        set_weights_from_sequence!(n.data)
    else
        n.data.site = site
        n.data.weights = zeros(Float64, Q)
        _fetch_weight_down!(n, site, method)
    end
end

function set_weights_from_sequence!(x::AState{Q}) where Q
    x.weights = zeros(Float64, Q)
    x.weights[x.sequence[x.site]] = 1
    return nothing
end

function _fetch_weight_down!(n, site, method)
    # in this one we know n is not a leaf node
    if method.sequence_model_type == :profile
        fetch_weight_down_profile!(n, site, method.sequence_model)
    elseif method.sequence_model_type == :ArDCA
        fetch_weight_down_ardca!()
    else
        throw(ArgumentError("Unrecognized sequence model $(method.sequence_model_type)"))
    end
end

function fetch_weight_down_profile!(n::TreeNode, site, model)
    for c in children(n)
        if isempty(c.data.weights) || c.site != site
            throw(ErrorException("Child $(c.label) of $n not initialized."))
        end
        P = P(model, branch_length(c))'
        n.data.weights .+= P * c.data.weights
    end
    return nothing
end
