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
strategy_marginal = ASR.ASRMethod(; joint = false, ML = true)
strategy_joint = ASR.ASRMethod(; joint=true, ML = true)

@testset "Felsenstein reconstruction" begin
    @test isapprox(
        exp(ASR.tree_likelihood!(tree, model, strategy_marginal)), lk_fel_marginal;
        rtol=1e-5
    )
    # @test isapprox(
    #     exp(ASR.tree_likelihood!(tree, model, strategy_joint)), lk_fel_joint;
    #     rtol=1e-4
    # )

    t, res = ASR.infer_ancestral(tree, model, strategy_marginal)
    rec_marginal = map(["R", "I1", "I2", "I3"]) do label
        ASR.intvec_to_sequence(t[label].data.sequence; alphabet=:nt)
    end
    @test rec_marginal == ["C", "A", "G", "C"]
    # @test res.max_likelihood ≈ res.likelihood # not working yet
    # @test isapprox(exp(res.max_likelihood), lk_fel_joint; rtol=1e-4)
end

# @testset "Felsenstein Bousseau" begin
#     t = ASR.bousseau_alg(tree, model, strategy_marginal)
#     lk = log(lk_fel_marginal)
#     @test all(≈(lk; rtol = 1e-4), map(ASR.bousseau_likelihood, nodes(t)))
# end
