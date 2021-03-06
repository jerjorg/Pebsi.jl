using Test

import Pebsi.Polynomials: sample_simplex,barytocart
import Pebsi.QuadraticIntegration: order_vertices!,quadval_vertex, edge_intersects

@testset "QuadraticIntegration" begin
    @testset "order_vertices" begin 

    simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
    order_vertices!(simplex)
    simplex == [0.0 0.5 0.5; 1.0 0.0 1.0]

    simplex = reduce(hcat,[[0,0],[0,1],[1,0]])
    order_vertices!(simplex)
    simplex == reduce(hcat,[[0,0],[1,0],[0,1]])

    end

    @testset "quadval_vertex" begin
        dim = 1
        deg = 2
        simplex = [0 1]
        barypts = sample_simplex(dim,deg)
        cartpts = barytocart(barypts,simplex)
        
        coeffs = [1,0,-1]
        @test quadval_vertex(coeffs) == -Inf
        
        coeffs = [1,0,1]
        @test quadval_vertex(coeffs) == 0.5
        
        coeffs = [0,0,0]
        @test isnan(quadval_vertex(coeffs))
        
        coeffs = [-1,0,-1]
        @test quadval_vertex(coeffs) == -0.5
        
        coeffs = [-1,-2,-4]
        @test quadval_vertex(coeffs) == 0
    end

    @testset "edge_intersects" begin
        dim = 1
        deg = 2
        simplex = [0 1]
        barypts = sample_simplex(dim,deg)
        cartpts = barytocart(barypts,simplex)

        a = 1
        b = 2
        c = -a+2b
        coeffs = [a,b,c]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [0,0,0.2]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [2,0,-3]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.4494897427831779]

        coeffs = [1,0,-1]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.5]

        coeffs = [1,0,1]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [-1,0,-1]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [-1,-2,-4]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [-1,3,-4]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.1603574565909282, 0.566915270681799]

        coeffs = [2,3,-4]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.6403882032022076]

        coeffs = [2,4,0]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [2,1,0]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [2,-4,8]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [4,3,2]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == []

        coeffs = [4,-3,1e-11]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.4000000000002667, 0.9999999999983334]

        coeffs = [1e-13,-3,1e-13]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [1.665334536937679e-14, 0.9999999999999831]

        coeffs = [-0.6591549430918953, -0.15915494309189532, 0.3408450569081046]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.6591549430918954]
    end
end