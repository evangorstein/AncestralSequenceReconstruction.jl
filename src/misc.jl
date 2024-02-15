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

function entropy(X::AbstractVector)
    if !isapprox(sum(X), 1, rtol=1e-5)
        error("Probability vector does not sum to one - got $(sum(X))")
    end

    return sum(X) do x
        x == 0 ? 0 : -x*log(x)
    end
end

function generate_state_table(tree::Tree{AState{q}}, alphabet) where q
    alphabet = Alphabet(alphabet)
    header = vcat(
        ["Node", "Site", "State"],
        map(i -> "p_" * alphabet.string[i], 1:q)
    )

    n = length(nodes(tree)) - length(leaves(tree))
    L = length(first(nodes(tree)).data.pstates)
    tab = Matrix{Any}(undef, n*L+1, length(header))
    tab[1, :] .= header

    counter = 0
    for (counter, (pos, node)) in enumerate(Iterators.product(1:L, internals(tree)))
        tab[counter+1, 1] = label(node)
        tab[counter+1, 2] = pos
        tab[counter+1, 3] = alphabet.string[node.data.pstates[pos].c]
        tab[counter+1, 4:end] .= map(x -> @sprintf("%1.5f", x),node.data.pstates[pos].posterior)
    end
    tab
end
