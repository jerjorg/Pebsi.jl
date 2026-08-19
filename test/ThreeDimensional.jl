using Test
import Pebsi.EPMs: Ag_epm
using Pebsi.Mesh: ibz_borders
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!

# Nothing three-dimensional was covered anywhere in the suite, and it did not
# work: `ibz_borders` indexed the hull's points with a facet, but in 3D a facet
# arrives as its own coordinates rather than as indices into them, so building a
# mesh raised before any integration happened. Two dimensions never touched that
# branch, so the whole suite passed regardless.
#
# These are deliberately loose. They are here to catch 3D being broken outright,
# which is what happened, and not to assert an accuracy the calculation does not
# yet reach - see docs/tolerance-robustness.md for where 3D convergence actually
# stands.
@testset "ThreeDimensional" begin
    @testset "ibz_borders" begin
        borders,distfun = ibz_borders(Ag_epm.ibz)
        @test length(borders) > 0
        # Points in the columns, as ptface_mindist reads them, and spatial: three
        # rows for three dimensions.
        @test all(b -> size(b,1) == 3, borders)
        @test all(b -> size(b,2) >= 3, borders)
        # The distance function has to accept what the borders hand it.
        @test distfun([0.0,0.0,0.0], borders[1]) >= 0
    end

    @testset "band structure" begin
        ebs = init_bandstructure(Ag_epm)
        @test size(ebs.mesh.points,1) > 0
        calc_flbe!(Ag_epm,ebs)
        # Within a percent of the stored true values: enough to catch a mesh or an
        # integration that has gone wrong, without claiming convergence.
        @test isapprox(ebs.fermilevel, Ag_epm.fermilevel, rtol=1e-2)
        @test isapprox(ebs.bandenergy, Ag_epm.bandenergy, rtol=1e-2)
    end
end
