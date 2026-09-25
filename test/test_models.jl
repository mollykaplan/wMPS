#dense array of O₁ ⊗ O₂ ⊗ ⋯, with the index order of a TensorMap (out₁, …, outₙ, in₁, …, inₙ)
function tensor_product(Ms...)
    n = length(Ms)
    d = size(first(Ms), 1)
    arr = ones(ComplexF64, ntuple(_ -> d, 2n))
    for I in CartesianIndices(arr)
        arr[I] = prod(Ms[k][I[k], I[n + k]] for k in 1:n)
    end
    return arr
end

@testset "operator helpers match the minimal-MPO alphabet (cutoff = $cutoff)" for cutoff in (2, 3)
    _, mats, index = wMPS.boson_alphabet(ComplexF64, cutoff, 2, 2)
    O(p, q) = mats[index[(p, q)]]      #(a⁺)ᵖ aᑫ
    expected = [
        wMPS.a_pp => (O(2, 0),),
        wMPS.a_mm => (O(0, 2),),
        wMPS.a_ppmm_4 => (O(2, 2),),
        wMPS.a_ppmm_3m => (O(2, 1), O(0, 1)),
        wMPS.a_ppmm_3p => (O(1, 2), O(1, 0)),
        wMPS.a_ppmm_3pm => (O(1, 1), O(1, 1)),
        wMPS.a_ppmm_3ppmm => (O(2, 0), O(0, 2)),
        wMPS.a_ppmm_3mmpp => (O(0, 2), O(2, 0)),
        wMPS.a_ppmm_2mm => (O(2, 0), O(0, 1), O(0, 1)),
        wMPS.a_ppmm_2pm => (O(1, 0), O(1, 1), O(0, 1)),
        wMPS.a_ppmm_2mp => (O(0, 1), O(1, 1), O(1, 0)),
        wMPS.a_ppmm_2pp => (O(0, 2), O(1, 0), O(1, 0)),
        wMPS.a_ppmm => (O(1, 0), O(1, 0), O(0, 1), O(0, 1)),
        wMPS.a_mmpp => (O(0, 1), O(0, 1), O(1, 0), O(1, 0)),
    ]
    for (op, factors) in expected
        @test convert(Array, op(ComplexF64; cutoff)) ≈ tensor_product(factors...)
    end
end

#LL_D8 is left out of the default suite, since building its MPO (bond dimension 2466) is very slow
@testset "constructor call forms: $model" for model in (SLOW ? (LL_D6, LL_D8, quad_model, quad_model_D8) : (LL_D6, quad_model, quad_model_D8))
    @test model() isa InfiniteMPOHamiltonian
    @test model(InfiniteChain(1)) isa InfiniteMPOHamiltonian
    @test model(ComplexF64, InfiniteChain(1)) isa InfiniteMPOHamiltonian
    @test model(Trivial) isa InfiniteMPOHamiltonian
    @test model(FiniteChain(10)) isa FiniteMPOHamiltonian
end

@testset "energies are real: $model" for model in (LL_D6, quad_model, quad_model_D8)
    @test isreal_energy(expectation_value(InfiniteMPS(ℂ^3, ℂ^4), model(; cutoff=2)))
end
if SLOW
    #LL_D8 on an infinite chain is too slow even for the slow tests
    @testset "energies are real: LL_D8 (finite chain)" begin
        L = 8
        @test isreal_energy(expectation_value(FiniteMPS(L, ℂ^3, ℂ^2), LL_D8(FiniteChain(L); cutoff=2)))
    end
end

#regression values: catches accidental changes in how @mpoham generates the terms
@testset "@mpoham bond dimensions" begin
    @test dim(left_virtualspace(LL_D6(; cutoff=2), 1)) == 562
    SLOW && @test dim(left_virtualspace(LL_D8(; cutoff=2), 1)) == 2466
end
