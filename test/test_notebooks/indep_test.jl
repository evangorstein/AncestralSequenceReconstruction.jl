### A Pluto.jl notebook ###
# v0.19.27

using Markdown
using InteractiveUtils

# ╔═╡ 9c926766-480c-11ee-03c9-334e2e5bc139
begin
	using Revise
	using Pkg; Pkg.activate("../")
	using AncestralSequenceReconstruction
	using BackwardCoalescent
	using Chain
	using Plots
	using StatsBase
	using SubstitutionModels
	using TreeAlgs
	using TreeTools
end

# ╔═╡ 3185ad1e-7584-4a3a-b4b6-5a95d4f9b78f
begin
	n = 15
	L = 20
	Troot = 1
	N = Troot / 2
	
	# ps = [.6, .2, .1, .1]
	ps = @chain [0.5, 0.5, 1e-16, 1e-16] _/sum(_)
	ev_model = GTR(1., 1., 1., 1., 1., ps...)

	asr_method = ASR.ASRMethod(;
		alphabet = :nt,
		L,
		sequence_model_type = :profile,
		evolution_model = ev_model,
		ML = true,
	)
end

# ╔═╡ 20cce8b2-ab06-4260-8d4d-f7ca1099473e


# ╔═╡ 46cefa9e-ed27-4f68-8887-4890cc4f3d1c
md"## Tests"

# ╔═╡ d7c8cf12-17d6-4d78-b32c-050885ada9f6
md"# Functions"

# ╔═╡ 9f942fea-0660-4ce1-9a77-ca4a11901dd3
function reconstruct(tree)
	tc = copy(tree)
	# remove sequence data on internal nodes
	for n in internals(tc)
		n.data = ASR.AState{4}()
	end
	# ASR
	ASR.infer_ancestral!(tc, asr_method)
	return tc
end

# ╔═╡ 8e1c0cbe-f0aa-4773-bf64-d28cb6a63623
begin
	function find_closest_leaf(n::TreeNode)
		find_closest_leaf(n, nothing)
	end
	# origin: node from which function was called (do not go back)
	# above: true if origin = parent(n), false otherwise
	function find_closest_leaf(n::TreeNode, origin::Union{Nothing, TreeNode})
		if isleaf(n)
			return label(n), 0.
		end
		
		(lab, d) = (nothing, Inf) # label and distance of closest leaf
		# check all children
		for c in Iterators.filter(!=(origin), children(n))
			(lab_c, d_c) = find_closest_leaf(c, n)
			if d_c + branch_length(c) < d
				lab = lab_c
				d = d_c + branch_length(c)
			end
		end
		# check ancestor
		if !isroot(n) && ancestor(n) != origin
			(lab_a, d_a) = find_closest_leaf(ancestor(n), n)
			if d_a + branch_length(n) < d
				d = d_a + branch_length(n)
				lab = lab_a
			end
		end

		return lab, d
	end
end

# ╔═╡ 4dd3b98e-6a62-4f34-bab9-009a2f170008
function compare(t_real, t_asr)
	@assert sort(map(label, t_real)) == sort(map(label, t_asr)) "trees do not share nodes"

	return map(internals(t_asr)) do n
		nr = t_real[label(n)] # corresponding node from the real tree
		predvec = n.data.sequence .== nr.data.sequence
		(
			label = label(n),
			predvec = predvec,
			no_errors = sum(.!predvec),
			closest_leaf = find_closest_leaf(n)[2],
		)
	end
end

# ╔═╡ 3f0266c9-d235-41d8-a06b-cc03b05cd051
function compare_fitch(t_real, t_asr)
	@assert sort(map(label, t_real)) == sort(map(label, t_asr)) "trees do not share nodes"

	return map(internals(t_asr)) do n
		nr = t_real[label(n)] # corresponding node from the real tree
		predvec = n.data.reconstructed_sequence .== nr.data.sequence
		(
			label = label(n),
			predvec = predvec,
			no_errors = sum(.!predvec),
			closest_leaf = find_closest_leaf(n)[2],
		)
	end
end

# ╔═╡ 8b6f0fd4-ddbf-4aaa-88eb-6a70d15f98a1
function miscdata_to_astate(dat::TreeTools.MiscData; key = :seq)
	out = AState{4}()
	out.sequence = ASR.sequence_to_int(string(dat[key]); alphabet = :nt)
	return out
end

# ╔═╡ 74df120d-0289-4c1d-a769-a8caae08eb1a
function generate_tree(;n=n, L=L)
	coalescent = KingmanCoalescent(n, N)
	t = @chain genealogy(coalescent) convert(Tree{TreeTools.MiscData}, _)
	Evolve.evolve!(t, L; model = ev_model)
	
	tree = convert(Tree{ASR.AState{4}}, t)
	for n in nodes(tree)
		n.data = miscdata_to_astate(t[label(n)].data)
	end
	
	return tree
end

# ╔═╡ 5c4622c9-6e9e-4954-8d5f-bd464fd82274
sim_results = let
	mapreduce(vcat, 1:100) do _
		treal = generate_tree()
		tasr = reconstruct(treal)
		compare(treal, tasr)
	end
end

# ╔═╡ 81f705a0-983d-4850-84e6-5712427854d5
sim_results_fitch = let
	mapreduce(vcat, 1:100) do _
		treal = generate_tree()
		sequences = Dict(label(n) => n.data.sequence for n in leaves(treal))
		tasr = Fitch.fitch(treal, sequences)
		compare_fitch(treal, tasr)
	end
end

# ╔═╡ 382681a3-cac7-452f-b715-940e4317b922
let
	p = plot()

	dleaf = map(x -> x.closest_leaf, sim_results)
	no_errors = map(x -> x.no_errors, sim_results)
	scatter(dleaf, no_errors / L, label = "", markerstrokewidth=0)
	
	scatter!(
		map(x -> x.closest_leaf, sim_results_fitch), 
		map(x -> x.no_errors, sim_results_fitch)/L;
		label = "", markerstrokewidth=0, alpha = .7
	)
	
	drange = range(0, maximum(dleaf), length=10)
	# plot!(drange, map(t -> .75 * (1-exp(-t)), drange), label="")
	hline!()
end

# ╔═╡ e7818797-f6a5-427d-afa7-206f0f488788
t = generate_tree(;n = 3, L = 1)

# ╔═╡ 55a774fe-5554-44d4-b922-ee33762df6bb
begin
	sequences = Dict(label(n) => n.data.sequence for n in leaves(t))
	tf = Fitch.fitch(t, sequences)
	map(n -> n.data.reconstructed_sequence, internals(tf))
	
	C = compare_fitch(t, tf)
	no_errors = map(x -> x.no_errors, C)
end

# ╔═╡ 214746b4-f04f-4227-8179-025292895c47
sequences

# ╔═╡ f9870ad6-0eb0-4150-b61b-5df42e35f903
@chain t["internal_1"]  POTleaves  map(label, _)

# ╔═╡ 3f530202-335d-44d9-94d2-f91a5b1f8beb


# ╔═╡ Cell order:
# ╠═9c926766-480c-11ee-03c9-334e2e5bc139
# ╠═3185ad1e-7584-4a3a-b4b6-5a95d4f9b78f
# ╠═5c4622c9-6e9e-4954-8d5f-bd464fd82274
# ╠═81f705a0-983d-4850-84e6-5712427854d5
# ╠═20cce8b2-ab06-4260-8d4d-f7ca1099473e
# ╠═382681a3-cac7-452f-b715-940e4317b922
# ╠═46cefa9e-ed27-4f68-8887-4890cc4f3d1c
# ╠═e7818797-f6a5-427d-afa7-206f0f488788
# ╠═55a774fe-5554-44d4-b922-ee33762df6bb
# ╠═f9870ad6-0eb0-4150-b61b-5df42e35f903
# ╠═214746b4-f04f-4227-8179-025292895c47
# ╟─d7c8cf12-17d6-4d78-b32c-050885ada9f6
# ╠═74df120d-0289-4c1d-a769-a8caae08eb1a
# ╠═9f942fea-0660-4ce1-9a77-ca4a11901dd3
# ╠═4dd3b98e-6a62-4f34-bab9-009a2f170008
# ╠═3f0266c9-d235-41d8-a06b-cc03b05cd051
# ╠═8e1c0cbe-f0aa-4773-bf64-d28cb6a63623
# ╠═8b6f0fd4-ddbf-4aaa-88eb-6a70d15f98a1
# ╠═3f530202-335d-44d9-94d2-f91a5b1f8beb
