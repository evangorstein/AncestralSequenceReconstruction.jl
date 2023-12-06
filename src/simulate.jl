module Simulate

using AncestralSequenceReconstruction
using FASTX
using TreeTools

"""
    evolve(tree::Tree, model; alphabet, leaves_fasta, internals_fasta)

Simulate sequences along `tree` using `model`.
Return the tree, the leaf sequences and the internal sequences.
Write output to alignments `leaves_fasta` and `internals_fasta`.
`tree` can be any type of tree: it will be converted to `Tree{AState}` inside.
"""
function evolve(
    tree::Tree, model::EvolutionModel{q};
    alphabet=ASR.default_alphabet(q), leaves_fasta = "", internals_fasta = "", root=nothing,
) where q
    L = length(model)
    tc = convert(Tree{ASR.AState{q}}, tree)
    foreach(n -> n.data = ASR.AState{q}(;L), nodes(tc))
    leaf_sequences, internal_sequences = evolve!(tc, model; alphabet, root)

    # write sequences to fasta if asked
    if !isempty(leaves_fasta)
        FASTAWriter(open(leaves_fasta, "w")) do writer
            for (name, seq) in leaf_sequences
                write(writer, FASTARecord(name, seq))
            end
        end
    end
    if !isempty(internals_fasta)
        FASTAWriter(open(internals_fasta, "w")) do writer
            for (name, seq) in internal_sequences
                write(writer, FASTARecord(name, seq))
            end
        end
    end

    return (leaf_sequences=leaf_sequences, internal_sequences=internal_sequences, tree=tc)
end

function evolve!(
    tree::Tree{ASR.AState{q}}, model::EvolutionModel;
    alphabet=:aa, root=nothing,
) where q
    # simulation
    strategy = ASR.ASRMethod(;joint=true, ML=false, alphabet, optimize_branch_length=false)
    for pos in ASR.ordering(model)
        ASR.set_pos(pos) # set global var pos
        ASR.reset_state!(tree, pos)
        # set transition matrices for all branches
        ASR.set_transition_matrix!(tree, model, pos)
        # set set state from transition matrix
        # down likelihood should be 1 since never initialized
        if isnothing(root)
            ASR.set_states!(tree, pos, strategy)
        else
            tree.root.data.pstates[pos].c = root[pos]
            foreach(c -> ASR.set_state!(c, root[pos], pos, strategy), children(tree.root))
        end
        foreach(n -> n.data.sequence[pos] = n.data.pstates[pos].c, nodes(tree))
    end

    # collect sequences
    leaf_sequences = map(leaves(tree)) do n
        label(n) => ASR.intvec_to_sequence(n.data.sequence; alphabet)
    end
    internal_sequences = map(internals(tree)) do n
        label(n) => ASR.intvec_to_sequence(n.data.sequence; alphabet)
    end

    return leaf_sequences, internal_sequences
end

end
