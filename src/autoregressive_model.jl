@kwdef mutable struct AutoRegressiveModel{q} <: EvolutionModel{q}
    arnet::ArDCA.ArNet
    μ :: Float64 = 1.
    with_code :: Bool = false
    genetic_code :: Matrix{Float64} = zeros(Float64, q, q)
    function AutoRegressiveModel{q}(arnet, μ, with_code, genetic_code) where q
        @assert length(arnet.p0) == q "Expected arnet with $q states"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !with_code || q == length(AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        return new{q}(arnet, μ, with_code, genetic_code)
    end
end

function AutoRegressiveModel(arnet::ArDCA.ArNet; kwargs...)
    q = length(arnet.p0)
    return AutoRegressiveModel{q}(; arnet, kwargs...)
end

length(model::AutoRegressiveModel) = length(model.arnet.H) + 1

ordering(model::AutoRegressiveModel) = model.arnet.idxperm

#=
########## set_π ##########
=#
function set_π!(astate::AState{q}, model::AutoRegressiveModel{q}, pos::Int) where q
    idxperm = model.arnet.idxperm
    ar_pos = findfirst(==(pos), idxperm)
    # special case of first ordering position
    if ar_pos == 1
        for (a, x) in enumerate(model.arnet.p0)
            astate.pstates[pos].weights.π[a] = x
        end
        return nothing
    end

    J = model.arnet.J[ar_pos-1]
    H = model.arnet.H[ar_pos-1]

    local_field = copy(H)
    for ar_i in 1:(ar_pos-1)
        b = astate.pstates[idxperm[ar_i]].c
        if isnothing(b)
            b = astate.sequence[idxperm[ar_i]]
        end
        isnothing(b) && error("Sequence not yet reconstructed at site $(idxperm[ar_i])")
        for a in 1:q
            local_field[a] += J[a, b, ar_i]
        end
    end
    ArDCA.softmax!(local_field)

    for (a, x) in enumerate(local_field)
        astate.pstates[pos].weights.π[a] = x
    end
end
