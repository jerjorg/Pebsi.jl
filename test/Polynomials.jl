using Test

using Pebsi.Polynomials: sample_simplex, bernstein_basis, getpoly_coeffs, eval_poly
using Pebsi.Geometry: barytocart
using PyCall: pyimport
sympy=pyimport("sympy")
using SymPy: symbols, Sym

@testset "Polynomials" begin
    @testset "bernstein_basis" begin
        (s,t,u,v)=symbols("s,t,u,v")
        dim = 1
        deg = 1
        @test bernstein_basis([s,t],dim,deg) == Sym[1.0*s, 1.0*t]

        dim = 1
        deg = 2
        @test bernstein_basis([s,t],dim,deg) == Sym[1.0*s^2, 2.0*s*t, 1.0*t^2]

        dim = 1
        deg = 3
        @test bernstein_basis([s,t],dim,deg) == Sym[1.0*s^3, 3.0*s^2*t, 3.0*s*t^2, 1.0*t^3]

        dim = 2
        deg = 1
        @test bernstein_basis([s,t,u],dim,deg) == Sym[1.0*s, 1.0*t, 1.0*u]

        dim = 2
        deg = 2
        @test bernstein_basis([s,t,u],dim,deg) == Sym[1.0*s^2, 2.0*s*t, 1.0*t^2, 2.0*s*u, 2.0*t*u, 1.0*u^2]

        dim = 2
        deg = 3
        @test bernstein_basis([s,t,u],dim,deg) == Sym[1.0*s^3, 3.0*s^2*t, 3.0*s*t^2, 1.0*t^3, 3.0*s^2*u, 6.0*s*t*u, 3.0*t^2*u, 3.0*s*u^2, 3.0*t*u^2, 1.0*u^3]

        dim = 3
        deg = 1
        @test bernstein_basis([s,t,u,v],dim,deg) == Sym[1.0*s, 1.0*t, 1.0*u, 1.0*v]

        dim = 3
        deg = 2
        @test bernstein_basis([s,t,u,v],dim,deg) == Sym[1.0*s^2, 2.0*s*t, 1.0*t^2, 2.0*s*u, 2.0*t*u, 1.0*u^2, 2.0*s*v, 2.0*t*v, 2.0*u*v, 1.0*v^2]

        dim = 3
        deg = 3
        @test bernstein_basis([s,t,u,v],dim,deg) == Sym[1.0*s^3, 3.0*s^2*t, 3.0*s*t^2, 1.0*t^3, 3.0*s^2*u, 6.0*s*t*u, 3.0*t^2*u, 3.0*s*u^2, 3.0*t*u^2, 1.0*u^3, 3.0*s^2*v, 6.0*s*t*v, 3.0*t^2*v, 6.0*s*u*v, 6.0*t*u*v, 3.0*u^2*v, 3.0*s*v^2, 3.0*t*v^2, 3.0*u*v^2, 1.0*v^3]
    end

    @testset "barytocart" begin
        barypt = [1,0,0]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        @test barytocart(barypt,simplex) == simplex[:,1]

        barypt = [0,1,0]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        @test barytocart(barypt,simplex) == simplex[:,2]

        barypt = [0,0,1]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        @test barytocart(barypt,simplex) == simplex[:,3]
    end

    @testset "eval_poly" begin
        simplex_bpts = [1.0 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
        values = [0.4, 0.5, 0.3, -0.4, -0.1, -0.2]
        dim = 2
        deg = 2
        coeffs = getpoly_coeffs(values,simplex_bpts,dim,deg)

        @test [mapslices(x->eval_poly(x,coeffs,dim,deg),simplex_bpts,dims=1)...] ≈ values
    end
end
