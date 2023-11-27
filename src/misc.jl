#=
Need every node of tree to have `seq` filled in
=#
function reconstruction_likelihood(tree::Tree{<:AState}, model::EvolutionModel)
    llk = sum(nodes(tree; skiproot=true)) do n
        log_transition_probability(
            data(ancestor(n)).sequence, data(n).sequence, branch_length(n), model
        )
    end
    llk += log_probability(data(tree.root).sequence, model)
    return llk
end
