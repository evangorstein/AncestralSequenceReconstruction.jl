module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

import Base: length

using DelimitedFiles
using FASTX
using LinearAlgebra
using StatsBase
using StaticArrays
using TreeTools

include("constants.jl")

include("objects.jl")
export ASRMethod

include("evolutionary_models.jl")
export EvolutionModel
export ProfileModel, JukesCantor

include("sequences.jl")

include("main.jl")
export infer_ancestral

end
