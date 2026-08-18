# The strategy selectors are enums rather than bare integers. This pins the new
# API: the numeric values still match the ones used in the published work, named
# values are accepted, and a bare integer is now rejected rather than silently
# selecting a branch.

using Test
using Pebsi.Defaults
using Pebsi.EPMs: m11
using Pebsi.QuadraticIntegration: init_bandstructure

@testset "StrategyEnums" begin

    @testset "numeric values are unchanged" begin
        @test Int(fl_bisection) == 1
        @test Int(fl_chandrupatla) == 2
        @test Int(refine_most_error) == 1
        @test Int(refine_largest_fraction) == 7
        @test Int(sample_center) == 1
        @test Int(sample_edge_midpoints) == 2
        @test Int(sample_adaptive) == 3
        @test Int(neighbors_closest) == 1
        @test Int(neighbors_inside) == 3
        @test Int(stop_total_error) == 1
        @test Int(stop_kpoint_target) == 4
    end

    @testset "every branch has a named value" begin
        @test length(instances(RefineMethod)) == 7
        @test length(instances(StopCriterion)) == 4
        @test length(instances(SampleMethod)) == 3
        @test length(instances(NeighborMethod)) == 3
        @test length(instances(FermiLevelMethod)) == 2
    end

    @testset "defaults are named values" begin
        @test def_refine_method == refine_largest_fraction
        @test def_stop_criterion == stop_kpoint_target
        @test def_sample_method == sample_edge_midpoints
        @test def_neighbor_method == neighbors_surrounding
        @test def_fermilevel_method == fl_chandrupatla
    end

    @testset "constructor accepts named values, rejects integers" begin
        ebs = init_bandstructure(m11; refine_method=refine_most_error,
                                 stop_criterion=stop_total_error)
        @test ebs.refine_method == refine_most_error
        @test ebs.stop_criterion == stop_total_error

        # Breaking change: this used to be how the method was selected. The
        # keyword's type annotation rejects it outright rather than converting,
        # so an old script fails loudly instead of quietly picking a branch.
        @test_throws TypeError init_bandstructure(m11; refine_method=1)
        @test_throws TypeError init_bandstructure(m11; stop_criterion=4)
    end
end
