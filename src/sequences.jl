const AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
const NT_ALPHABET = "ACGT"

aa_alphabet_names = (:aa, :AA, :aminoacids, :amino_acids)
nt_alphabet_names = (:nt, :nucleotide, :dna)

function alphabet_string(alphabet)
    return if alphabet in aa_alphabet_names
        AA_ALPHABET
    elseif alphabet in nt_alphabet_names
        NT_ALPHABET
    else
        unknown_alphabet_error(alphabet)
    end
end
alphabet_size(alphabet) = length(alphabet_string(alphabet))
function default_alphabet(q::Int)
    return if q == 21
        :aa
    elseif q == 4
        :nt
    else
        error("Not default alphabet for q=$q")
    end
end

"""
    alphabet_mapping(s::AbstractString)

`Dict(i => c for (i,c) in enumerate(s))`.
"""
alphabet_mapping(s::AbstractString) = Dict(c => i for (i, c) in enumerate(s))

const AA_MAPPING = alphabet_mapping(AA_ALPHABET)
const NT_MAPPING = alphabet_mapping(NT_ALPHABET)

function unknown_alphabet_error(a)
    throw(ArgumentError("""
        Incorrect alphabet type `$a`.
        Choose from `$aa_alphabet_names` or `$nt_alphabet_names`.
    """))
end

function sequence_to_intvec(s::AbstractString; alphabet = :aa)
    return if alphabet in aa_alphabet_names
        map(c -> AA_MAPPING[Char(c)], collect(s))
    elseif alphabet in nt_alphabet_names
        map(c -> NT_MAPPING[Char(c)], collect(s))
    else
        unknown_alphabet_error(alphabet)
    end
end
sequence_to_intvec(s::AbstractVector{<:Integer}; kwargs...) = s

function intvec_to_sequence(X::AbstractVector; alphabet=:aa)
    amap = alphabet_string(alphabet)
    return map(x -> amap[x], X) |> String
end

"""
    fasta_to_tree!(tree::Tree{AState}, fastafile::AbstractString)

Add sequences of `fastafile` to nodes of `tree`.
"""
function fasta_to_tree!(
    tree::Tree{AState{q}}, fastafile::AbstractString;
    warn = true, default=missing, alphabet = :aa,
) where q
    all_headers_in_tree = true
    all_leaves_in_fasta = true

    reader = open(FASTA.Reader, fastafile)
    record = FASTA.Record()
    while !eof(reader)
        read!(reader, record)
        if in(identifier(record), tree)
            seq = sequence_to_intvec(sequence(record); alphabet)
            if maximum(seq) > q
                error("""
                    $(typeof(Tree)) with $q states, found $(maximum(seq)) in sequence
                    Problem with alphabet?
                """)
            end
            tree[identifier(record)].data.sequence .= seq
        else
            all_headers_in_tree = false
        end
    end
    close(reader)

    for n in leaves(tree)
        if isempty(n.data.sequence)
            all_leaves_in_fasta = false
            break
        end
    end
    !all_leaves_in_fasta && @warn "Not all leaves had a corresponding sequence \
        in the alignment (file: $fastafile)."
    !all_headers_in_tree && @warn "Some sequence headers in the alignment are \
        not found in the tree (file: $fastafile)."
    return nothing
end

"""
    sequences_to_tree!(tree::Tree{<:AState}, seqmap; alphabet=:aa, safe)

Iterating `seqmap` should yield pairs `label => sequence`.
"""
function sequences_to_tree!(
    tree::Tree{AState{q}}, seqmap;
    alphabet=:aa, safe=true,
) where q
    for (label, seq) in seqmap
        if safe && !isleaf(tree[label])
            error("Cannot assign an observed sequence to internal node. Use `safe=false`?")
        end
        tree[label].data = AState{q}(;
            L = length(seq), sequence = sequence_to_intvec(seq; alphabet)
        )
    end
    if any(n -> !hassequence(n.data), leaves(tree))
        @warn "Somes leaves do not have sequences"
    end
    return nothing
end

"""
    initialize_tree(tree::Tree, seqmap; alphabet=:aa)
"""
function initialize_tree(tree::Tree, seqmap; alphabet=:aa)
    L = first(seqmap)[2] |> length
    q = alphabet_size(alphabet)
    tree = convert(Tree{AState{q}}, tree)
    foreach(n -> n.data = AState{q}(;L), nodes(tree))
    sequences_to_tree!(tree, seqmap; alphabet)
    return tree
end
