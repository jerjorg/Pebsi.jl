using test

import Pebsi.QuadraticIntegration: order_vertices

@testset "QuadraticIntegration" begin
    @testset "order_vertices" begin 

    simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
    order_vertices!(simplex)
    simplex == [0.0 0.5 0.5; 1.0 0.0 1.0]

    simplex = reduce(hcat,[[0,0],[0,1],[1,0]])
    order_vertices!(simplex)
    simplex == reduce(hcat,[[0,0],[1,0],[0,1]])

    end
end