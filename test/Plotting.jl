# Smoke tests for the Plotting module.
#
# These deliberately assert very little about the result: the point is that each
# entry point loads and runs at all. Plotting has broken three times in ways no
# numerical test could catch - reading `plt` at module scope, assuming the
# matplotlib colour cycle is made of strings, and reading it again in `__init__`
# where PyPlot has not initialised Python yet. All three were load- or first-call
# failures, which is exactly what this file covers.
#
# ENV["MPLBACKEND"] is set in runtests.jl, before anything loads PyPlot.

using Test
using PyPlot: PyObject, subplots, figure, subplot, close
using Pebsi.Plotting: meshplot, contourplot, bezplot, bezcurve_plot, polygonplot,
    plot_bandstructure, fermicurve_plot, default_colors
using Pebsi.Mesh: ibz_init_mesh
using Pebsi.EPMs: m11, Al_epm

@testset "Plotting" begin

    @testset "default_colors" begin
        c = default_colors()
        @test !isempty(c)
        @test c[1] !== nothing          # indexable, whatever the element type
        @test default_colors() === c    # cached, not refetched
    end

    @testset "meshplot" begin
        fig, ax = subplots()
        @test meshplot([0 1 1 0; 0 0 1 1], ax) isa PyObject
        close(fig)

        fig = figure()
        ax3 = subplot(projection="3d")
        @test meshplot([0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1], ax3) isa PyObject
        close(fig)

        mesh = ibz_init_mesh(m11.ibz, 3)
        fig, ax = subplots()
        @test meshplot(mesh, ax) isa PyObject
        close(fig)
    end

    @testset "contourplot" begin
        bezpts = [0. 0.5 1. 0.5 1. 1.; 0. 0. 0. 0.5 0.5 1.; -1 0.1 -0.2 1.2 1.1 -0.2]
        fig, ax = subplots()
        @test contourplot(bezpts, ax, padded=false) isa PyObject
        @test contourplot(bezpts, ax, padded=false, filled=true, ndiv=20,
                          colors=["red","black"]) isa PyObject
        close(fig)
    end

    @testset "bezplot" begin
        fig, ax = subplots()
        @test bezplot([0. 0.5 1.; -0.2 1.3 -0.5], ax) isa PyObject
        close(fig)
    end

    @testset "bezcurve_plot" begin
        @test bezcurve_plot([0.0 0.0 1.0; 1.0 1/3 0.0], [1.0, 1.5, 1.0]) isa PyObject
        close("all")
    end

    @testset "polygonplot" begin
        @test polygonplot([0 1 1 0; 0 0 1 1]) isa PyObject
        @test polygonplot([[0 1 1; 0 0 1], [0 1 0; 0 1 1]]) isa PyObject
        close("all")
    end

    @testset "fermicurve_plot" begin
        @test fermicurve_plot(m11, ntri=3, ndivs=5) isa PyObject
        close("all")
    end

    @testset "plot_bandstructure" begin
        # Al_epm rather than m11: seekpath returns high-symmetry points in 3D, so
        # despite accepting `epm-model2D` this fails on a 2D model with a 2x2
        # times 3-vector DimensionMismatch. Kept small - the defaults expand 1000
        # plane waves over 10 sheets.
        @test plot_bandstructure(Al_epm; expansion_size=50, sheets=2,
                                 kpoint_dist=0.5) isa PyObject
        close("all")
    end
end
