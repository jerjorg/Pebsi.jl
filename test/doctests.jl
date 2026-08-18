# Run the doctests as part of the test suite.
#
# They previously ran only inside the documentation build, where a failure is
# reported but does not fail the job. That is how the 2D border bug survived:
# `Pkg.test()` was green while a real code path was broken, and the only thing
# exercising it was a doctest nobody was gating on.

using Documenter, Test, Pebsi

@testset "doctests" begin
    doctest(Pebsi; manual=false)
end
