### A Pluto.jl notebook ###
# v0.19.29

using Markdown
using InteractiveUtils

# ╔═╡ cdb17bb0-69d7-11ee-12ea-8fe7d76c9f65
begin
	using Revise
	using Pkg; Pkg.activate("..")
	using AncestralSequenceReconstruction
	using TreeTools
	using Test
end	

# ╔═╡ c9fbaac0-bc77-40b8-8706-f38819b0525c
begin
	L = 1
	q = 4
	TT() = ASR.AState{q}(; L)
end

# ╔═╡ 04d459c0-c160-4a57-92aa-58cd109b5cff
leaf_sequences = Dict("A" => [1], "B" => [1], "C" => [2], "D" => [1])

# ╔═╡ 6dc76304-9604-4b22-be9b-34b8d901dadf
tree = begin
	nwk = "(((A,B)I1,C)I2,D)R;"
	tree = parse_newick_string(nwk; node_data_type = TT)
	foreach(n -> branch_length!(n, 1.), nodes(tree; skiproot=true))
	ASR.initialize_tree(tree, leaf_sequences; alphabet=:nt)
end

# ╔═╡ 32b5f992-e857-4409-859a-dded0d5b4070
model = ASR.JukesCantor(L);

# ╔═╡ 953aa5a6-ad44-4f26-b5c4-aa275ca8d603
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

# ╔═╡ 94195889-2bed-4501-bbd7-369b14ff6ac8


# ╔═╡ 92a6ae7b-5af1-4f96-be3b-12c18717c25d
begin
	@test T2["I2"].data.pstates[1].weights.u ==  T1["I2"].data.pstates[1].weights.u
	@test T2["I2"].data.pstates[1].weights.v !=  T1["I2"].data.pstates[1].weights.v

	ASR.update_neighbours!(T1["I1"]; anc = true)

	@test T2["I2"].data.pstates[1].weights.u ==  T1["I2"].data.pstates[1].weights.u
	@test T2["I2"].data.pstates[1].weights.v ==  T1["I2"].data.pstates[1].weights.v
	@test T2["I2"].data.pstates[1].weights.Zv[] ==  T1["I2"].data.pstates[1].weights.Zv[]
end

# ╔═╡ 9e6b8ff3-7ab9-486c-8aa3-03c3bbfff2b5
T2["I2"].data.pstates[1].weights.v

# ╔═╡ 1340ae44-3749-4711-a5a5-2031f3c5c730
T1["I2"].data.pstates[1].weights.v

# ╔═╡ c595d1da-df38-4e21-a743-e7ab8301b6b6
ASR.update_neighbours!(T1["I1"]; anc = false)

# ╔═╡ Cell order:
# ╠═cdb17bb0-69d7-11ee-12ea-8fe7d76c9f65
# ╠═c9fbaac0-bc77-40b8-8706-f38819b0525c
# ╠═04d459c0-c160-4a57-92aa-58cd109b5cff
# ╠═6dc76304-9604-4b22-be9b-34b8d901dadf
# ╠═32b5f992-e857-4409-859a-dded0d5b4070
# ╠═953aa5a6-ad44-4f26-b5c4-aa275ca8d603
# ╠═94195889-2bed-4501-bbd7-369b14ff6ac8
# ╠═92a6ae7b-5af1-4f96-be3b-12c18717c25d
# ╠═9e6b8ff3-7ab9-486c-8aa3-03c3bbfff2b5
# ╠═1340ae44-3749-4711-a5a5-2031f3c5c730
# ╠═c595d1da-df38-4e21-a743-e7ab8301b6b6
