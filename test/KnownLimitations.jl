# Failures that are known, understood, and deliberately not fixed. Asserting them
# keeps the behaviour visible and pins it: if someone fixes one of these, the
# corresponding test fails and prompts them to update this file and the docs.

using Test
using Pebsi.EPMs: m31, m11
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!
using Pebsi.Plotting: plot_bandstructure

@testset "KnownLimitations" begin

    @testset "m31 integrates" begin
        # m31 used to fail here: `split_bezsurf` handed back a patch it could not
        # subdivide, with three curve intersections and no formula to integrate
        # it. Patches that small are now integrated by their sign, so this
        # completes. Asserted against the true band energy rather than a literal,
        # since the value moves with the platform.
        ebs = init_bandstructure(m31)
        calc_flbe!(m31, ebs)
        @test abs(ebs.bandenergy - m31.bandenergy) < 1e-2
    end

    @testset "plot_bandstructure needs a 3D model" begin
        # seekpath returns high-symmetry points in 3D, so this fails on a 2D model
        # with a 2x2 times 3-vector mismatch despite the signature accepting EPM2D.
        @test_throws DimensionMismatch plot_bandstructure(m11; expansion_size=50,
                                                          sheets=2, kpoint_dist=0.5)
    end
end
