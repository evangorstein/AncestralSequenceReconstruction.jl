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

I use a $\pi$ that is $\simeq 1/3$ for $A, I$ and $V$, and $\varepsilon$ for the other symbols.  Then the tree is 
```
|--------- A
|
|--------- I
```

Given the genetic code, there is a good probability that the ancestor is a $V$, since it is a transition symbol between $A$ and $I$. Without the genetic code, the probability of $V$ at the root should be $\varepsilon$ for short branch lengths 
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
	t = .1
	tree = parse_newick_string("(A:$(t),B:$(t))R;")
	sequences = Dict("A" => "A", "B" => "I")
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

# ╔═╡ 11493665-95df-46ba-9235-3cf841e1a112
alphabet = ASR.Alphabet(:AA)

# ╔═╡ 44a60bbb-ea0e-4dad-8aba-35cae73a56d5
let
	@info "With genetic code"
	method = ASRMethod(ML=false, optimize_branch_length=true, verbosity=0)
	
	map(1:100) do _
		t = infer_ancestral(copy(tree), model_wcode, method)
		alphabet.string[t.root.data.sequence[1]]
		# map(branch_length, leaves(t))
	end |> countmap
end

# ╔═╡ bae564d9-1ad8-44cb-8329-6ba81265fbf5
let
	@info "Without genetic code"
	method = ASRMethod(ML=false, optimize_branch_length=true, verbosity=0)
	
	map(1:100) do _
		t = infer_ancestral(copy(tree), model_nocode, method)
		alphabet.string[t.root.data.sequence[1]]
		# map(branch_length, leaves(t))
	end |> countmap
end

# ╔═╡ Cell order:
# ╠═7d1dbb58-256a-11ef-26e2-49bb2ecbcb99
# ╟─db56f90b-fe6e-41b3-9f34-2c207a6000e5
# ╠═b3e3afd7-e229-49a1-8593-9d8fc876856a
# ╠═5e7e234b-4b6c-4fe7-83f2-fdb3790c5eb4
# ╠═d2e6c146-940a-4af4-b655-260ea96fc9a1
# ╠═ba8f57e5-00c9-42ab-9128-82acbeb96d82
# ╠═0d0b0f37-72b3-4501-9b92-cf09a576c09f
# ╠═11493665-95df-46ba-9235-3cf841e1a112
# ╠═44a60bbb-ea0e-4dad-8aba-35cae73a56d5
# ╠═bae564d9-1ad8-44cb-8329-6ba81265fbf5
