using AncestralSequenceReconstruction
using Test
using TreeTools

@testset "AncestralSequenceReconstruction.jl" begin
    # Example in Felsenstein's "Inferring phylogenies" in section 16.4
    include("Felsenstein/test.jl")
end
