dir = dirname(@__FILE__)

dir = "Felsenstein/"
using AncestralSequenceReconstruction
using Test
using TreeTools

L = 1
q = 4

tree = read_tree(dir * "/tree.16.4.nwk"; node_data_type=ASR.AState{q})
ASR.fasta_to_tree!(tree, dir * "/alignment.16.4.fasta"; alphabet=:nt)

lk_fel_marginal = 0.0000124065
lk_fel_joint = 0.0000041594
joint_reconstruction = Dict("I1" => 'A', "I2" => 'G', "I3" => 'G')

model = ASR.JukesCantor(L)
strategy_marginal = ASR.ASRMethod(; joint = false)
strategy_joint = ASR.ASRMethod(; joint=true)

@testset "Felsenstein reconstruction" begin
    @test isapprox(
        exp(ASR.tree_likelihood!(tree, model, strategy_marginal)), lk_fel_marginal;
        rtol=1e-5
    )
    @test isapprox(
        exp(ASR.tree_likelihood!(tree, model, strategy_joint)), lk_fel_joint;
        rtol=1e-4
    )

    t, res = ASR.infer_ancestral(tree, model, strategy_joint)
    reconstruction = map(["R", "I1", "I2", "I3"]) do label
        ASR.intvec_to_sequence(t[label].data.sequence; alphabet=:nt)
    end
    @test reconstruction == ["G", "A", "G", "G"]
    @test res.max_likelihood ≈ res.likelihood
    @test isapprox(exp(res.max_likelihood), lk_fel_joint; rtol=1e-4)
end

@testset "Felsenstein Pupko" begin
    t = ASR.pupko_alg(tree, model, strategy_marginal)
    L = log(lk_fel_marginal)
    @test all(≈(L; rtol = 1e-4), map(ASR.pupko_likelihood, nodes(t)))
end
