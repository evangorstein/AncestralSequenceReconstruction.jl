"""
    abstract type EvolutionModel{q}

Describes the dynamics of sequences for a state space (alphabet) of size `q`.
Concrete types of `EvolutionModel` should implement `set_π!` and `set_transition_matrix!`.
"""
abstract type EvolutionModel{q} end

###################################################################################
#################################### NECESSARY ####################################
###################################################################################

"""
    set_π!(astate::AState, model::EvolutionModel, pos::Int)

Set equilibrium frequencies of for ancestral state `astate` at site `pos` using `model`.
"""
function set_π! end
"""
    set_π!(tree::Tree, model::EvolutionModel, pos::Int)

Set equilibrium frequencies at site `pos` for all nodes.
"""
function set_π!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_π!(n.data, model, pos)
    end
    return nothing
end

"""
    set_transition_matrix!(T::Matrix{Float64}, model::EvolutionModel, t, pos, π)
    set_transition_matrix!(astate::AState, model::EvolutionModel, t, pos)

Set transition matrix to the input ancestral state, using branch length `t`.
In the first form, store the output in matrix `T`.
In the second form, store in  `astate.pstates[pos].weights.T`.

*Note*: In the first form, the `π` argument (equilibrium frequencies) is needed if
we are using ArDCA.
"""
function set_transition_matrix! end
"""
    set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)

Set the transition matrix for all branches in `tree` at seqeunce position `pos`.
"""
function set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_transition_matrix!(n.data, model, branch_length(n), pos)
    end
    return nothing
end


###################################################################################
###################################### USEFUL #####################################
###################################################################################


"""
    set_transition_rate_matrix!(Q::Matrix{Float64}, model::EvolutionModel, t, pos, π)

Set transition rate matrix in place, in matrix `Q`.
`π` is the expected equilibrium probability (needed only for AR model).
"""
function set_transition_rate_matrix! end

"""
    transition_probability(old::Int, new::Int, model::EvolutionaryModel, t, pos, π)
"""
function transition_probability end

"""
    transition_matrix(model::EvolutionModel, t, pos[, π])

Return the transition matrix for `model` at position `pos` and branch length `t`.
"""
function transition_matrix(model::EvolutionModel{q}, t, pos, π=ones(Float64, q)/q) where q
    T = zeros(Float64, q, q)
    set_transition_matrix!(T, model, t, pos, π)
    return T
end
"""
    transition_rate_matrix(model::EvolutionModel, pos)

Return the transition rate matrix for `model` at position `pos`.
"""
function transition_rate_matrix(model::EvolutionModel{q}, pos, π=ones(Float64, q)/q) where q
    Q = zeros(Float64, q, q)
    set_transition_rate_matrix!(Q, model, pos, π)
    return Q
end




