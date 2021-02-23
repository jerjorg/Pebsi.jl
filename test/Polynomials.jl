using Test

import Pebsi.Polynomials: sample_simplex, bernstein_basis, barytocart, 
    getpoly_coeffs, eval_poly, shadow_size, simplex_size, bezsimplex_size

import PyCall: pyimport
sympy=pyimport("sympy")
import SymPy: symbols, Sym

import QHull: chull

@testset "Polynomials" begin
    @testset "sample_simplex" begin
        deg = 1
        dim = 1
        @test sample_simplex(dim,deg) ≈ [1.0 0.0; 0.0 1.0]

        deg = 2
        dim = 1
        @test sample_simplex(dim,deg) ≈ [1.0 0.5 0.0; 0.0 0.5 1.0]

        deg = 3
        dim = 1
        @test sample_simplex(dim,deg) ≈ [1.0 0.6666666666666666 0.3333333333333333 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0]

        deg = 1
        dim = 2
        @test sample_simplex(dim,deg) ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

        deg = 2
        dim = 2
        @test sample_simplex(dim,deg) ≈ [1.0 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]

        deg = 3
        dim = 2
        @test sample_simplex(dim,deg) ≈ [1.0 0.6666666666666666 0.3333333333333333 0.0 0.6666666666666666 0.3333333333333333 0.0 0.3333333333333333 0.0 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0 0.0 0.3333333333333333 0.6666666666666666 0.0 0.3333333333333333 0.0; 0.0 0.0 0.0 0.0 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666 1.0]

        deg = 1
        dim = 3
        @test sample_simplex(dim,deg) ≈ [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0.0; 0.0 0.0 0.0 1.0]

        deg = 2
        dim = 3
        @test sample_simplex(dim,deg) ≈ [1.0 0.5 0.0 0.5 0.0 0.0 0.5 0.0 0.0 0.0; 0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0]
        
        deg = 3
        dim = 3
        @test sample_simplex(dim,deg) ≈ [1.0 0.6666666666666666 0.3333333333333333 0.0 0.6666666666666666 0.3333333333333333 0.0 0.3333333333333333 0.0 0.0 0.6666666666666666 0.3333333333333333 0.0 0.3333333333333333 0.0 0.0 0.3333333333333333 0.0 0.0 0.0; 0.0 0.3333333333333333 0.6666666666666666 1.0 0.0 0.3333333333333333 0.6666666666666666 0.0 0.3333333333333333 0.0 0.0 0.3333333333333333 0.6666666666666666 0.0 0.3333333333333333 0.0 0.0 0.3333333333333333 0.0 0.0; 0.0 0.0 0.0 0.0 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666 1.0 0.0 0.0 0.0 0.3333333333333333 0.3333333333333333 0.6666666666666666 0.0 0.0 0.3333333333333333 0.0; 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.3333333333333333 0.6666666666666666 0.6666666666666666 0.6666666666666666 1.0]
    end

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

    @testset "simplex_size" begin
        simplex = [0 0 1; 0 1 0]
        @test simplex_size(simplex) ≈ chull(Array(simplex')).volume

        simplex = [0.0 0.5 0.0; 1.0 0.0 0.0]
        @test simplex_size(simplex) ≈ chull(Array(simplex')).volume
    end

    @testset "shadow_size" begin
        coeffs = [0.4, 0.5, 0.3, -0.2, -0.1, -0.3]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        val = -0.3
        @test shadow_size(coeffs,simplex,val) == 0

        val = 0.5
        @test shadow_size(coeffs,simplex,val) ≈ chull(Array(simplex')).volume

        coeffs = [0.4, 0.5, 0.3, -0.2, -0.1, -0.3, 0.7, -0.6, 0.9, -0.7]
        simplex = [0.0 0.5 0.5 0.0; 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0]
        val = -0.7
        @test shadow_size(coeffs,simplex,val) == 0

        val = 0.9
        @test shadow_size(coeffs,simplex,val) ≈ chull(Array(simplex')).volume
    end

    @testset "bezsimplex_size" begin
        
        coeffs = [0.45695660389445203, 0.46891799288429503, 0.4825655582645329, -0.21218601852805546, -0.18566890981787773, -0.28153999982153577]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        @test bezsimplex_size(coeffs,simplex,100) ≈ 0.030376884453158788

        coeffs = [0.45695660389445203, -0.28153999982153577, -0.319970723890622]
        simplex = [0.0 0.5 0.0; 1.0 0.0 0.0]
        @test bezsimplex_size(coeffs,simplex,100) ≈ -0.012046176651475476

        coeffs = [0.4, 0.5, 0.3, -0.2, -0.1, -0.3]
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        val = -0.3
        @test bezsimplex_size(coeffs,simplex,val) == 0
    
        val = 0.5
        @test bezsimplex_size(coeffs,simplex,val) ≈ 0.025
    
        coeffs = [0.4, 0.5, 0.3, -0.2, -0.1, -0.3, 0.7, -0.6, 0.9, -0.7]
        simplex = [0.0 0.5 0.5 0.0; 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0]
        val = -0.7
        @test bezsimplex_size(coeffs,simplex,val) == 0
    
        val = 0.9
        bezsimplex_size(coeffs,simplex,val)
    
    end
end
