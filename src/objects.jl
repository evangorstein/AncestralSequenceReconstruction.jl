#######################################################################################
################################## Position weights ###################################
#######################################################################################

mutable struct BranchWeights{q}
    π :: Vector{Float64} # eq. probabiltiy of each state
    u :: Vector{Float64} # up-likelihood: probability of data excluding the subtree
    v :: Vector{Float64} # down-likelihood: probability of data below the subtree
    Zu :: Array{Float64, 0} # log-normalization for u --> u*exp(Zu) are the actual likelihoods
    Zv :: Array{Float64, 0} # log-normalization for v // dim 0 array is a trick for mutability
    T :: Matrix{Float64} # propagator matrix to this state, given branch length to ancestor : T[a,b] = T[a-->b] = T[b|a,t]
    c :: Vector{Int} # character state
    function BranchWeights{q}(π, u, v, Zu, Zv, T, c) where q
        @assert isapprox(sum(π), 1) "Probabilities must sum to one - got $(sum(π))"
        @assert all(r -> sum(r)≈1, eachrow(T)) "Rows of transition matrix should sum to 1"
        @assert length(π) == q "Expected frequency vector of dimension $q, got $π"
        @assert length(u) == q "Expected weights vector of dimension $q, got $u"
        @assert length(v) == q "Expected weights vector of dimension $q, got $v"
        @assert length(c) == q "Expected character state vector of dimension $q, got $c"
        @assert size(T,1) == size(T,2) == q "Expected transition matrix of dimension $q, got $T"
        return new{q}(π, u, v, Zu, Zv, T, c)
    end
end
function BranchWeights{q}(π) where q
    return BranchWeights{q}(
        π,
        ones(Float64, q),
        ones(Float64, q),
        fill(0.),
        fill(0.),
        diagm(ones(Float64, q)),
        zeros(Int, q),
    )
end

BranchWeights{q}() where q = BranchWeights{q}(ones(Float64, q)/q)

function Base.copy(W::BranchWeights{q}) where q
    return BranchWeights{q}([copy(getproperty(W, f)) for f in propertynames(W)]...)
end

function reset_weights!(W::BranchWeights{q}) where q
    for a in 1:q
        W.u[a] = 1
        W.v[a] = 1
        W.c[a] = 0
        # foreach(b -> W.T[a,b] = 0, 1:q)
        # W.T[a,a] = 1
    end
    W.Zu[] = 0.
    W.Zv[] = 0.

    return nothing
end

function reset_up_likelihood!(W::BranchWeights{q}) where q
    foreach(a -> W.u[a] = 1, 1:q)
    W.Zu[] = 0.
    return nothing
end
reset_up_likelihood!(n::TreeNode, pos) = reset_up_likelihood!(n.data.pstates[pos].weights)

function reset_down_likelihood!(W::BranchWeights{q}) where q
    foreach(a -> W.v[a] = 1, 1:q)
    W.Zv[] = 0.
    return nothing
end
function reset_down_likelihood!(n::TreeNode, pos)
    reset_down_likelihood!(n.data.pstates[pos].weights)
end

function normalize!(W::BranchWeights)
    Zv = sum(W.v)
    Zu = sum(W.u)
    # check for issues
    if Zv == 0 || Zu == 0
        error("Found likelihood weight equal to 0 - model may not be able to accomodate for data")
    end

    for i in eachindex(W.v)
        W.u[i] = W.u[i]/Zu
        W.v[i] = W.v[i]/Zv
    end
    W.Zu[] += log(Zu)
    W.Zv[] += log(Zv)

    return nothing
end
normalize_weights!(n::TreeNode, pos::Int) = normalize!(n.data.pstates[pos].weights)

#######################################################################################
#################################### Position state ###################################
#######################################################################################

@kwdef mutable struct PosState{q}
    pos::Int = 0
    c :: Union{Nothing, Int} = nothing # current state at this position
    lk::Float64 = 0.
    posterior :: Vector{Float64} = LinearAlgebra.normalize!(ones(Float64, q), 1)
    weights::BranchWeights{q} = BranchWeights{q}() # weights for the alg.
end

function Base.copy(pstate::PosState{q}) where q
    return PosState{q}(;
        pos=pstate.pos,
        c = pstate.c,
        lk = pstate.lk,
        posterior = copy(pstate.posterior),
        weights = copy(pstate.weights),
    )
end

function reset_state!(pstate::PosState{q}) where q
    pstate.c = nothing
    pstate.lk = 0
    pstate.posterior = LinearAlgebra.normalize!(ones(Float64, q), 1)
    reset_weights!(pstate.weights)
    return nothing
end

#######################################################################################
################################### Ancestral state ###################################
#######################################################################################

@kwdef struct AState{q} <: TreeNodeData
    L::Int = 1

    # state of the node during the algorithm
    pstates :: Vector{PosState{q}} = [PosState{q}(; pos=i) for i in 1:L]

    # final result: whole sequence at this node
    sequence :: Vector{Union{Nothing, Int}} = Vector{Nothing}(undef, L) # length L

    function AState{q}(L, pstates, sequence) where q
        @assert length(pstates) == length(sequence) == L """
            Incorrect dimensions: expected sequence of length $L, got $sequence and $pstates
        """
        @assert all(x -> x[2].pos == x[1], enumerate(pstates))
        return new{q}(L, pstates, sequence)
    end
end

function Base.copy(state::AState{q}) where q
    return AState{q}(;
        L = state.L,
        pstates = map(copy, state.pstates),
        sequence = copy(state.sequence),
    )
end

reset_state!(state::AState, pos::Int) = reset_state!(state.pstates[pos])
reset_state!(tree::Tree, pos::Int) = foreach(n -> reset_state!(n.data, pos), nodes(tree))
function reset_state!(node::TreeNode{<:AState})
    !isleaf(node) && data(node).sequence .= nothing
    for pos in 1:data(node).L
        reset_state!(data(node), pos)
    end
    return nothing
end

reconstructed_positions(state::AState) = findall(!isnothing, state.sequence)
is_reconstructed(state::AState, pos::Int) = !isnothing(state.sequence[pos])
hassequence(state::AState{q}) where q = all(i -> is_reconstructed(state, i), 1:state.L)


function Base.show(io::IO, ::MIME"text/plain", state::AState{q}) where q
    if !get(io, :compact, false)
        println(io, "Ancestral state (L: $(state.L), q: $q)")
        println(io, "Sequence $(state.sequence)")
    end
    return nothing
end
function Base.show(io::IO, state::AState)
    print(io, "$(typeof(state)) - \
     $(length(reconstructed_positions(state))) reconstructed positions")
    return nothing
end

#######################################################################################
####################################### Alphabet ######################################
#######################################################################################


const _AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
const _NT_ALPHABET_NOGAP = "ACGT"
const _SPIN_ALPHABET = "01"


_alphabet_mapping(s::AbstractString) = Dict(c => i for (i, c) in enumerate(s))

@kwdef struct Alphabet
    string::String
    mapping::Dict{Char, Int} = _alphabet_mapping(string)
end
Alphabet(s::AbstractString) = Alphabet(; string = s, mapping = _alphabet_mapping(s))
Alphabet(a::Alphabet) = a
function Alphabet(mapping::AbstractDict{Char, Int})
    str = Vector{Char}(undef, length(mapping))
    for (c, i) in mapping
        str[i] = c
    end
    return Alphabet(;string = prod(str), mapping)
end
function Alphabet(rev_mapping::AbstractDict{Int, Char})
    mapping = Dict{Char, Int}(c => i for (i,c) in rev_mapping)
    return Alphabet(mapping)
end

const aa_alphabet = Alphabet(_AA_ALPHABET)
const aa_alphabet_names = (:aa, :AA, :aminoacids, :amino_acids)

const nt_alphabet = Alphabet(_NT_ALPHABET_NOGAP)
const nt_alphabet_names = (:nt, :nucleotide, :dna)

const spin_alphabet = Alphabet(_SPIN_ALPHABET)
const sping_alphabet_names = (:spin,)

# Default alphabet from symbol
function Alphabet(alphabet::Symbol)
    return if alphabet in aa_alphabet_names
        aa_alphabet
    elseif alphabet in nt_alphabet_names
        nt_alphabet
    elseif alphabet in sping_alphabet_names
        spin_alphabet
    else
        unknown_alphabet_error(alphabet)
    end
end

Base.convert(::Type{Alphabet}, x::Symbol) = Alphabet(x)

Base.length(a::Alphabet) = length(a.string)

# Default alphabet for given size q
function default_alphabet(q::Int)
    return if q == 21
        aa_alphabet
    elseif q == 4
        nt_alphabet
    elseif q == 2
        spin_alphabet
    else
        error("Not default alphabet for q=$q")
    end
end

function unknown_alphabet_error(a)
    throw(ArgumentError("""
        Unrecognized alphabet name `$a`.
        Choose from `$aa_alphabet_names`, `$nt_alphabet_names`, `$sping_alphabet_names`, or provide a string, or construct with `Alphabet`.
    """))
end

# from string to `Vector{Int}`
function sequence_to_intvec(s::AbstractString; alphabet = :aa)
    return sequence_to_intvec(s, Alphabet(alphabet))
end
function sequence_to_intvec(s::AbstractString, alphabet::Alphabet)
    return map(c -> alphabet.mapping[Char(c)], collect(s))
end
sequence_to_intvec(s::AbstractVector{<:Integer}; kwargs...) = s

# from `Vector{Int}` to string
function intvec_to_sequence(X::AbstractVector; alphabet=:aa)
    return intvec_to_sequence(X, Alphabet(alphabet))
end
function intvec_to_sequence(X::AbstractVector, alphabet::Alphabet)
    return map(x -> alphabet.string[x], X) |> String
end


#######################################################################################
###################################### ASR Method #####################################
#######################################################################################

"""
    ASRMethod

- `joint::Bool`: joint or marginal inference. Default `false`.
- `ML::Bool`: maximum likelihood, or sampling. Default `false`.
- `alphabet :: ASR.Alphabet`: alphabet used to map from integers to sequences. Default `:aa`.
- `verbosity :: Int`: verbosity level. Default 0.
- `optimize_branch_length`: Optimize the branch lengths of the tree according to the model.
  Default `false`.
- `repetitions :: Int`: Number of repetitions for the reconstruction.
  Should be set to 1 for the ML reconstruction.
  For Bayesian reconstruction (*i.e.* `ML=false`), this allows for sampling of likely
  ancestors.
  If higher than 1, the `infer_ancestral` function will return an array of sequence
  mappings, one for each reconstruction.
  An array of output fasta files should also be provided.
"""
@kwdef mutable struct ASRMethod
    joint::Bool = true
    ML::Bool = false
    alphabet::Symbol = :aa
    alphabet::Alphabet = aa_alphabet
    verbosity::Int = 0
    optimize_branch_length::Bool = false
    optimize_branch_length_cycles::Int = 3
    repetitions::Int = 1
end
