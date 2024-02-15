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
BRANCH_LWR_BOUND(L) = log(L+2) - log(L+1)
BRANCH_UPR_BOUND(L) = log(L+2)

