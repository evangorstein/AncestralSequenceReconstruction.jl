module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

import Base: copy, length, convert

using Accessors
using ArDCA
using DelimitedFiles
using FASTX
using LinearAlgebra
using NLopt
using Printf
using StatsBase
using TreeTools

include("objects.jl")
export ASRMethod

include("constants.jl")

include("evolutionary_models.jl")
export EvolutionModel
include("profile_model.jl")
export ProfileModel, JukesCantor
include("autoregressive_model.jl")
export AutoRegressiveModel

include("sequences.jl")
include("felsenstein.jl") # core algorithm
include("reconstruction.jl")
export infer_ancestral

include("time_opt.jl")
include("misc.jl")
include("simulate.jl")

end
