####### MODELS #######

L, q = (3, 2)
Pmat = [
    1/3 1/2 1/4;
    2/3 1/2 3/4;
]
Pvec = [[1/3, 2/3], [1/2, 1/2], [1/4, 3/4]]
@testset "IndependentModel basics" begin
    I2 = ASR.IndependentModel(Pvec)
    @test length(I2.ordering) == L
    @test typeof(I2) == ASR.IndependentModel{q}

    I1 = ASR.IndependentModel(Pmat)
    @test length(I1.ordering) == L
    @test typeof(I1) == ASR.IndependentModel{q}


    @test I1.P == I2.P
    @test I2.ordering == I2.ordering

    @test_throws AssertionError ASR.IndependentModel(Pvec .+ [0.01*rand(q) for _ in 1:L]) # should not use rand but well...
    @test_throws AssertionError ASR.IndependentModel(Pmat .+ rand(q, L))

    JC = JukesCantor(1)
    @test JC.P[1] == 1/4 * ones(Float64, 4)
end
