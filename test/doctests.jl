# Run the doctests as part of the test suite.
#
# They previously ran only inside the documentation build, where a failure is
# reported but does not fail the job. That is how the 2D border bug survived:
# `Pkg.test()` was green while a real code path was broken, and the only thing
# exercising it was a doctest nobody was gating on.
#
# Restricted to the Julia version the documentation is built with. Seven of these
# doctests print full-precision floating point, and those digits differ between
# Julia versions because the BLAS and LLVM underneath differ - on 1.10 the same
# calculation ends elsewhere in the last few ulp. Gating every version on exact
# output makes the matrix flaky without adding any signal about the docs, which
# are only ever rendered from one version.

using Documenter, Test, Pebsi

const DOCS_JULIA_VERSION = v"1.11"

if VERSION >= DOCS_JULIA_VERSION
    @testset "doctests" begin
        doctest(Pebsi; manual=false)
    end
else
    @info "Skipping doctests on Julia $VERSION: exact floating-point output is " *
          "version-dependent. They are gated on Julia $DOCS_JULIA_VERSION and newer, " *
          "and in the documentation build."
end
