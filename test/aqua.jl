using Aqua, Test, Pebsi

@testset "method ambiguity" begin Aqua.test_ambiguities(Pebsi) end
Aqua.test_all(Pebsi, ambiguities=false)
