let pos::Int = 1
    global current_pos() = pos
    global increment_pos() = (pos += 1)
    global set_pos(i::Int) = (pos = i)
    global reset_pos() = set_pos(0)
end

let n::Int = 1
    global get_count() = n
    global inc_count() = (n += 1)
    global reset_count() = (n = 1)
end

let v::Int = 0
    global set_verbose(val) = (v=val)
    global verbose() = v
end

# using the idea (1 - e(-t)) ∼ (n+1) / (L+2) (fraction of mutated sites, with a pc.)
# for n = L and n = 0 we get the limits below
BRANCH_LWR_BOUND_BAYES(L) = log(L+2) - log(L+1)
BRANCH_UPR_BOUND_BAYES(L) = log(L+2) * 0.75 # this over-estimates since saturation occurs before -- would need to put long term eq. of model

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


############################################################################################

const aa_order = ['K', 'N', 'K', 'N', 'T', 'T', 'T', 'T', 'R', 'S', 'R', 'S', 'I', 'I', 'M', 'I', 'Q', 'H', 'Q', 'H', 'P', 'P', 'P', 'P', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'E', 'D', 'E', 'D', 'A', 'A', 'A', 'A', 'G', 'G', 'G', 'G', 'V', 'V', 'V', 'V', '*', 'Y', '*', 'Y', 'S', 'S', 'S', 'S', '*', 'C', 'W', 'C', 'L', 'F', 'L', 'F']
const nt_order = ['A', 'C', 'G', 'T']

const gencode_as_dict = let
    D = Dict()
    i = 1
    for a in nt_order, b in nt_order, c in nt_order
        D[prod([a, b, c])] = aa_order[i]
        i += 1
    end
    D
end

function _mut_codons(codon)
    X = mapreduce(vcat, 1:3) do i
        map(nt_order) do a
            c = collect(codon)
            c[i] = a
            prod(c)
        end
    end
    filter(!=(codon), X)
end

const gencode_as_mat = let
    alphabet = ASR.Alphabet(:aa)
    q = length(alphabet)
    R = zeros(Int, q, q)
    codon_list = [prod([a,b,c]) for a in nt_order for b in nt_order for c in nt_order]
    for codon in codon_list
        row = if haskey(alphabet.mapping, gencode_as_dict[codon])
             alphabet.mapping[gencode_as_dict[codon]]
        else
            continue
        end
        # @info alphabet.string[row] codon
        for mut_codon in _mut_codons(codon)
            col = if haskey(alphabet.mapping, gencode_as_dict[mut_codon])
                alphabet.mapping[gencode_as_dict[mut_codon]]
            else
                continue
            end
            # @info "--> $(alphabet.string[col]) / $(mut_codon)"
            R[row, col] += 1
        end
    end
    for a in 1:q
        R[a,a] = 0
    end
    R ./ mean(R)
    # R
end
