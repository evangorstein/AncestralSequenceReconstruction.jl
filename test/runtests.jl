using AncestralSequenceReconstruction
using Test
using TreeTools

@testset "AncestralSequenceReconstruction.jl" begin
    # Basic tests for evolution models
    include("basics/emodels.jl")
    # Example in Felsenstein's "Inferring phylogenies" in section 16.4
    include("Felsenstein/test.jl")
end
