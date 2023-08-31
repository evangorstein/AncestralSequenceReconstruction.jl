module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

using FASTX
using StatsBase
using SubstitutionModels
using TreeTools

include("objects.jl")
export AState

include("sequences.jl")


include("main.jl")

end
