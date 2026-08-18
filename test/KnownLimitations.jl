# Failures that are known, understood, and deliberately not fixed. Asserting them
# keeps the behaviour visible and pins it: if someone fixes one of these, the
# corresponding test fails and prompts them to update this file and the docs.

using Test
using Pebsi.EPMs: m31, m11
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!
using Pebsi.Plotting: plot_bandstructure

@testset "KnownLimitations" begin

    @testset "m31 integration" begin
        # `split_bezsurf` can hand back a surface with three intersections when a
        # triangle is too small to subdivide - the trigger is an area around 3e-9,
        # while `def_min_simplex_size` only catches 1e-12. Diagnosed in a comment
        # at the error site in `two_intersects_area_volume`. Deciding how such a
        # patch should contribute is a question about the integration method, so
        # it is recorded rather than papered over.
        ebs = init_bandstructure(m31)
        @test_throws ErrorException calc_flbe!(m31, ebs)
    end

    @testset "plot_bandstructure needs a 3D model" begin
        # seekpath returns high-symmetry points in 3D, so this fails on a 2D model
        # with a 2x2 times 3-vector mismatch despite the signature accepting EPM2D.
        @test_throws DimensionMismatch plot_bandstructure(m11; expansion_size=50,
                                                          sheets=2, kpoint_dist=0.5)
    end
end
