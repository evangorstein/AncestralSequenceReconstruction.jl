module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

import Base: copy, length

using DelimitedFiles
using FASTX
using LinearAlgebra
using StatsBase
using TreeTools

include("constants.jl")

include("objects.jl")
export ASRMethod

include("evolutionary_models.jl")
export EvolutionModel
export ProfileModel, JukesCantor

include("sequences.jl")

include("reconstruction.jl")
export infer_ancestral

include("time_opt.jl")

end
