### A Pluto.jl notebook ###
# v0.19.28

using Markdown
using InteractiveUtils

# ╔═╡ cc01e746-5d19-11ee-0cb1-7767701b2acc
begin
	using Pkg; Pkg.activate(".")
	using BioSequences
	using Chain
	using Combinatorics
	using DelimitedFiles
end

# ╔═╡ 7898bc3c-fe8f-4b76-8880-cc2369f98108
const dna_alphabet = @chain alphabet(DNA) filter(!isambiguous, _) filter(!isgap, _)

# ╔═╡ 93c8b8e3-1028-4eea-881d-e144414f7d66
const aa_alphabet = begin
	aa_mapping = "-ACDEFGHIKLMNPQRSTVWY"
	map(x -> convert(AminoAcid, x), collect(aa_mapping))
end

# ╔═╡ 04e49d56-dfb2-42d6-a66f-cc53542af619
reverse_aa_mapping = Dict(a => i for (i,a) in enumerate(aa_alphabet))

# ╔═╡ b4e90921-2bb9-4193-ae92-2d52215f1e6d
md"""
- `inverse_translate[aa]::Dict`: list of codons mapping to `aa`.
- `codon_mut_map::Dict`: for each codon `C`, to which amino-acid does it translate after all possible single mutations.
- `aa_mut_map::Matrix`: AAs indexed from 2 to 21 (1 is gap). Row `[i, j]`: how many ways are there to go from aa `i` to `j` with one mutation, considering all codons that can represent `i`. 
"""

# ╔═╡ 3f89fd91-1df0-4ead-b590-80eddf0f4dc4
const codons = @chain begin
	with_replacement_combinations(dna_alphabet, 3)
	map(permutations, _)
	Iterators.flatten
	unique
	map(x -> LongDNA{4}(x), _)
	collect
end

# ╔═╡ bdeae819-c48a-47fa-8124-1b043c5e6056
function single_mutants(codon)
	return map(1:3) do i
		a = codon[i]
		map(Iterators.filter(!=(a), dna_alphabet)) do b
			m = copy(collect(codon))
			m[i] = b
			LongDNA{4}(m)
		end
	end |> Iterators.flatten |> collect
end

# ╔═╡ 966b8b2a-b923-4baf-89a2-7ecc0b57ee0a
codon_mut_map = let
	M = Dict(cod => [] for cod in codons)
	for cod in codons, mcod in single_mutants(cod)
		push!(M[cod], translate(mcod)[1])
	end
	foreach(AAs -> filter!(aa -> in(aa, aa_alphabet), AAs), values(M))
	M
end

# ╔═╡ 27fa439f-5fd7-4459-8887-9ef9262adf6e
function build_inverse_translate()
	invT = Dict(aa => [] for aa in aa_alphabet)
	for cod in codons
		aa = translate(cod)[1]
		in(aa, aa_alphabet) && push!(invT[aa], cod)
	end
	invT
end

# ╔═╡ 26739707-08e0-4e87-82fc-9b4e3019f3ec
inverse_translate = build_inverse_translate();

# ╔═╡ 3508e4ac-4a2c-444b-93b0-d8cfda59d1c4
aa_mut_map = let
	M = zeros(Int, length(aa_alphabet), length(aa_alphabet))
	for (i, aa) in Iterators.filter(x -> !isgap(x[2]), enumerate(aa_alphabet))
		for codon in inverse_translate[aa], mut_aa in codon_mut_map[codon]
			M[i, reverse_aa_mapping[mut_aa]] += 1
		end
	end
	M
end

# ╔═╡ adc0fb01-bfeb-47f7-9de5-b8de444070c0
begin
	writedlm("aa_transition_rate_matrix.dat", aa_mut_map)
	write("aa_mapping.txt", aa_mapping)
end

# ╔═╡ Cell order:
# ╠═cc01e746-5d19-11ee-0cb1-7767701b2acc
# ╠═7898bc3c-fe8f-4b76-8880-cc2369f98108
# ╠═93c8b8e3-1028-4eea-881d-e144414f7d66
# ╠═04e49d56-dfb2-42d6-a66f-cc53542af619
# ╟─b4e90921-2bb9-4193-ae92-2d52215f1e6d
# ╠═3f89fd91-1df0-4ead-b590-80eddf0f4dc4
# ╠═26739707-08e0-4e87-82fc-9b4e3019f3ec
# ╠═966b8b2a-b923-4baf-89a2-7ecc0b57ee0a
# ╠═3508e4ac-4a2c-444b-93b0-d8cfda59d1c4
# ╠═adc0fb01-bfeb-47f7-9de5-b8de444070c0
# ╠═bdeae819-c48a-47fa-8124-1b043c5e6056
# ╠═27fa439f-5fd7-4459-8887-9ef9262adf6e
