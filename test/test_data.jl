#coefficient data and wavelet constants that everything else is built on

@testset "coefficient arrays" begin
    @test size(wMPS.Γ4_D6) == (9, 9, 9)
    @test size(wMPS.Γ4_D8) == (13, 13, 13)
    @test length(wMPS.K6) == 5
    @test length(wMPS.K8) == 7
    for arr in (wMPS.K6, wMPS.K8, wMPS.Γ4_D6, wMPS.Γ4_D8)
        @test eltype(arr) <: Real
        @test all(isfinite, arr)
    end
end

#all permutations of four elements
perms4(v) = [v[[i, j, k, l]] for i in 1:4 for j in 1:4 for k in 1:4 for l in 1:4 if allunique((i, j, k, l))]

#Γ4[a,b,c] is the overlap of wavelets at positions (0, a, b, c); as a function of the four
#positions it must be invariant under permutations, which LL_terms relies on
function max_permutation_asymmetry(Γ4)
    w = (size(Γ4, 1) - 1) ÷ 2
    function overlap(pos)
        offsets = pos[2:4] .- pos[1]
        all(abs.(offsets) .<= w) || return 0.0
        return Γ4[(offsets .+ (w + 1))...]
    end
    err = 0.0
    for I in CartesianIndices(Γ4)
        pos = [0; collect(Tuple(I)) .- (w + 1)]
        for p in perms4(pos)
            err = max(err, abs(overlap(p) - Γ4[I]))
        end
    end
    return err / maximum(abs, Γ4)
end

@testset "Γ4 permutation symmetry" begin
    @test max_permutation_asymmetry(wMPS.Γ4_D6) < 1e-14
    @test max_permutation_asymmetry(wMPS.Γ4_D8) < 1e-14
end

@testset "D6 filter and IWT matrices" begin
    @test sum(wMPS.h6) ≈ sqrt(2)
    @test sum(abs2, wMPS.h6) ≈ 1
    for A in (wMPS.A1, wMPS.A2, wMPS.A3)
        @test A' * A ≈ I atol = 1e-14
    end
end

@testset "scaling function table" begin
    @test length(wMPS.sxdata) == 101
    @test wMPS.sxdata[1] == 0
    @test wMPS.sxdata[end] == 0
    @test wMPS.sint == wMPS.sxdata[[21, 41, 61, 81]]      #s(1), …, s(4) with dx = 0.05
    @test wMPS.sxdata[1] + sum(wMPS.sint) ≈ 1 atol = 1e-10  #partition of unity at integers
end
