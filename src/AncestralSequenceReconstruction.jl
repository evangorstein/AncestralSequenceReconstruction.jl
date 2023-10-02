module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

import Base: length

using LinearAlgebra
using FASTX
using StatsBase
using StaticArrays
using TreeTools

include("constants.jl")

include("objects.jl")
export AState

include("evolutionary_models.jl")

include("sequences.jl")

include("main.jl")

end
