# Run the doctests as part of the test suite.
#
# They used to run only inside the documentation build, where a failure is
# reported but does not fail the job. That is how the 2D border bug survived:
# `Pkg.test()` was green while a real code path was broken, and the only thing
# exercising it was a doctest nobody was gating on.
#
# Gating these needs the doctests to be reproducible. Seven of them printed
# full-precision floating point, which differs between platforms and Julia
# versions because the BLAS and LLVM underneath differ, and they failed on Linux
# and Windows while passing on macOS/arm64. Those now round to ten digits, which
# is far beyond the observed spread of 1e-12 to 1e-13 and does not claim
# precision the arithmetic cannot reproduce.

using Documenter, Test, Pebsi

@testset "doctests" begin
    doctest(Pebsi; manual=false)
end
