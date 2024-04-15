using AncestralSequenceReconstruction
using ArDCA
using JLD2
using StatsBase
using Test
using TreeTools


@testset "AncestralSequenceReconstruction.jl" begin
    @testset "basics" begin
        # Basic tests for evolution models
        include("basics/emodels.jl")
    end
    @testset "Felsenstein" begin
        # Example in Felsenstein's "Inferring phylogenies" in section 16.4
        include("Felsenstein/test.jl")
    end
    @testset "time_opt" begin
        # Bousseau alg: update neighbours
        include("time_opt/test.jl")
    end
end
