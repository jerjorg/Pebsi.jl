using Test

using Pebsi.Geometry

@testset "Geometry" begin
    @testset "order_vertices" begin 
        simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
        order_vertices!(simplex)
        @test simplex == [0.0 0.5 0.5; 1.0 0.0 1.0]
  
        simplex = reduce(hcat,[[0,0],[0,1],[1,0]])
        order_vertices!(simplex)
        @test simplex == reduce(hcat,[[0,0],[1,0],[0,1]])
      end
end