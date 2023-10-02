dir = dirname(@__FILE__)

L = 1
q = 4

tree = read_tree(dir * "/tree.16.4.nwk"; node_data_type=ASR.AState{L,q})
ASR.fasta_to_tree!(tree, dir * "/alignment.16.4.fasta"; alphabet=:nt)

lk_fel_marginal = 0.0000124065
lk_fel_joint = 0.0000041594
joint_reconstruction = Dict("I1" => 'A', "I2" => 'G', "I3" => 'G')

model = ASR.JukesCantor(L)
strategy_marginal = ASR.ASRMethod(; joint = false)
strategy_joint = ASR.ASRMethod(; joint=true)

@testset "Felsenstein's example" begin
    @test isapprox(
        exp(ASR.tree_likelihood!(tree, model, strategy_marginal)), lk_fel_marginal;
        rtol=1e-5
    )
    @test isapprox(
        exp(ASR.tree_likelihood!(tree, model, strategy_joint)), lk_fel_joint;
        rtol=1e-4
    )
end
