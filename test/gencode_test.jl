### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ 7d1dbb58-256a-11ef-26e2-49bb2ecbcb99
begin
	using Revise
	using Pkg; 
	Pkg.activate(".")
	using AncestralSequenceReconstruction
	using LinearAlgebra
	using JLD2
	using StatsBase
	using TreeTools
end

# ╔═╡ db56f90b-fe6e-41b3-9f34-2c207a6000e5
md"""
	V (19) --> A (2): one mutation
	V (19) --> I (9): one mutation
	A --> I: two mutations, going through V or T (18)
"""

# ╔═╡ 5e7e234b-4b6c-4fe7-83f2-fdb3790c5eb4
normalize!(x) = (x .= x / sum(x))

# ╔═╡ b3e3afd7-e229-49a1-8593-9d8fc876856a
begin
	L = 1
	q = 21
	
	π = 1e-3 * ones(q)
	π[[2, 9, 19]] .= .3
	normalize!(π)
end

# ╔═╡ d2e6c146-940a-4af4-b655-260ea96fc9a1
begin
	model_nocode = ProfileModel([π])
	model_wcode = ProfileModel([π]; with_code=true)
end

# ╔═╡ ba8f57e5-00c9-42ab-9128-82acbeb96d82
tree = let
	t = .3
	tree = parse_newick_string("(A:$(t),B:$(t))R;")
	sequences = Dict("A" => "A", "B" => "C")
	ASR.initialize_tree(tree, sequences)
end

# ╔═╡ 0d0b0f37-72b3-4501-9b92-cf09a576c09f
begin
	delta(i::Int) = let
		p = zeros(Float64, q)
		p[i] = 1
		p
	end
	delta(a::Char) = delta(model_wcode.alphabet.mapping[a])
end

# ╔═╡ d6a1d203-c261-4a86-aa79-8449b809fc6e
sum(π)

# ╔═╡ 44a60bbb-ea0e-4dad-8aba-35cae73a56d5
let
	method = ASRMethod(ML=false, optimize_branch_length=true, verbosity=2)
	# t = copy(tree)
	# t = infer_ancestral(t, model_wcode, method)
	# let
	# 	w = t.root.data.pstates[1].weights
	# 	w.u .* w.v |> normalize!
	# end
	# t.root.data.sequence[1]

	# map(sum, eachcol(t.root.data.pstates[1].weights.T))
	# t.root.data.pstates[1].weights.T
	
	map(1:1) do _
		t = infer_ancestral(copy(tree), model_nocode, method)
		t.root.data.sequence[1]
		map(branch_length, leaves(t))
	end #|> countmap
end

# ╔═╡ 3d882973-ca5e-4834-be31-44fae8f2395f
Q = let
	Q = zeros(q,q)
	ASR.set_transition_matrix_gencode!(Q, Inf,  π)
end

# ╔═╡ Cell order:
# ╠═7d1dbb58-256a-11ef-26e2-49bb2ecbcb99
# ╠═b3e3afd7-e229-49a1-8593-9d8fc876856a
# ╟─db56f90b-fe6e-41b3-9f34-2c207a6000e5
# ╠═5e7e234b-4b6c-4fe7-83f2-fdb3790c5eb4
# ╠═d2e6c146-940a-4af4-b655-260ea96fc9a1
# ╠═ba8f57e5-00c9-42ab-9128-82acbeb96d82
# ╠═0d0b0f37-72b3-4501-9b92-cf09a576c09f
# ╠═d6a1d203-c261-4a86-aa79-8449b809fc6e
# ╠═44a60bbb-ea0e-4dad-8aba-35cae73a56d5
# ╠═3d882973-ca5e-4834-be31-44fae8f2395f
