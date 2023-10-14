# using Pkg; Pkg.activate("..")
using AncestralSequenceReconstruction
using TreeTools
using Test

#
L = 1
q = 4
TT() = ASR.AState{q}(; L)

# could be anything
leaf_sequences = Dict("A" => [1], "B" => [1], "C" => [2], "D" => [1])
model = ASR.JukesCantor(L);

# initial tree
tree = begin
    nwk = "(((A,B)I1,C)I2,D)R;"
    tree = parse_newick_string(nwk; node_data_type = TT)
    foreach(n -> branch_length!(n, 1.), nodes(tree; skiproot=true))
    ASR.initialize_tree(tree, leaf_sequences; alphabet=:nt)
end

# two trees
# T1 has the messages precomputed, and then we change a branch's length
# T2 is the same, but the likelihood is re-computed with the full alg. after
# the branch whose length is changed is I2 --> I1
T1, T2 = begin
    # T1: compute lk for tree, change branch after
    T1 = copy(tree)
    ASR.bousseau_alg!(T1, model)
    branch_length!(T1["I1"], 2.)
    ASR.set_transition_matrix!(T1["I1"].data.pstates[1], model, 2.)

    # T2: change branch and fully compute lk ~ truth
    T2 = copy(tree)
    branch_length!(T2["I1"], 2.)
    ASR.bousseau_alg!(T2, model)

    T1, T2
end

W1(label) = T1[label].data.pstates[1].weights
W2(label) = T2[label].data.pstates[1].weights

@testset "Update neighbours after length change" begin
    @testset "Ancestor" begin
        @test W1("I2").u ≈ W2("I2").u
        @test W1("I2").v != W2("I2").v # down-lk for I2 changed

        ASR.update_neighbours!(T1["I1"]; anc=true)

        @test W1("I2").u ≈ W2("I2").u
        @test W1("I2").v ≈ W2("I2").v
        @test W1("I2").Zv[] ≈ W2("I2").Zv[]
    end


    @testset "Sister" begin
        @test W1("C").u != W2("C").u
        @test W1("C").v ≈ W2("C").v

        ASR.update_neighbours!(T1["I1"]; sisters=true)

        @test W1("C").u ≈ W2("C").u
        @test W1("C").Zu[] ≈ W2("C").Zu[]
        @test W1("C").v ≈ W2("C").v
    end

    @testset "Children" begin
        for c in ["A", "B"]
            @test W1(c).u != W2(c).u
            @test W1(c).v ≈ W2(c).v
        end

        ASR.update_neighbours!(T1["I1"]; child=true)

        for c in ["A", "B"]
            @test W1(c).u ≈ W2(c).u
            @test W1(c).Zu[] ≈ W2(c).Zu[]
            @test W1(c).v ≈ W2(c).v
        end
    end

    @testset "Final" begin
        @test W1("I2").v ≈ W2("I2").v
        @test W1("C").u ≈ W2("C").u
        @test W1("A").u ≈ W2("A").u
        @test W1("B").u ≈ W2("B").u
    end
end
