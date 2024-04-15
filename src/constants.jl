let pos::Int = 1
    global current_pos() = pos
    global increment_pos() = (pos += 1)
    global set_pos(i::Int) = (pos = i)
    global reset_pos() = set_pos(0)
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
