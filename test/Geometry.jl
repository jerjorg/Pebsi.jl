using Test

using QHull: chull
using LinearAlgebra: cross,norm,dot
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

  @testset "ptface_mindist" begin
    tet = [0 1 0 0; 0 0 1 0; 0 0 0 1.]

    face = tet[:,[2,3,4]]
    pt = [1,0,0]
    @test ptface_mindist(pt,face) == 0
    
    face = tet[:,[1,2,3]]
    pt = [0.1,0.1,0]
    @test ptface_mindist(pt,face) == 0
    
    face = tet[:,[1,3,4]]
    pt = [0.0,0.1,0.1]
    @test ptface_mindist(pt,face) == 0
    
    face = tet[:,[1,3,4]]
    pt = [0.1,0.1,0.1]
    @test ptface_mindist(pt,face) == 0.1
    
    face = tet[:,[1,2,4]]
    pt = [1.2,0,0]
    @test ptface_mindist(pt,face) ≈ 0.2
    
    face = tet[:,[2,3,4]]
    ϵ = 0.0
    pt = [1/√9+ϵ,1/√9+ϵ,1/√9+ϵ]
    @test round(ptface_mindist(pt,face),digits=15) == ϵ
    
    face = tet[:,[2,3,4]]
    ϵ = 0.1
    pt = [1/√9+ϵ,1/√9+ϵ,1/√9+ϵ]
    @test ptface_mindist(pt,face) ≈ ϵ*√3
    
    face = tet[:,[1,2,4]]
    pt = [0.5,-1,0]
    @test ptface_mindist(pt,face) ≈ 1
    
    face = tet[:,[1,2,4]]
    pt = [0.1,0.1,0]
    @test ptface_mindist(pt,face) ≈ 0.1      
  end

  @testset "point_in_polygon" begin
    tet = [0 1 0 0; 0 0 1 0; 0 0 0 1.]
    face = tet[:,[2,3,4]]
    pt = [1,0,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,2,3]]
    pt = [0.1,0.1,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,3,4]]
    pt = [0.0,0.1,0.1]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,3,4]]
    pt = [0.1,0.1,0.1]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,2,4]]
    pt = [1.2,0,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == false
    
    face = tet[:,[2,3,4]]
    ϵ = 0.0
    pt = [1/√9+ϵ,1/√9+ϵ,1/√9+ϵ]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[2,3,4]]
    ϵ = 0.1
    pt = [1/√9+ϵ,1/√9+ϵ,1/√9+ϵ]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,2,4]]
    pt = [0.5,-1,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,2,4]]
    pt = [0.1,0.1,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == true
    
    face = tet[:,[1,2,4]]
    pt = [1,0,1]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == false
    
    face = tet[:,[1,2,4]]
    pt = [0,0,1.3]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == false

    tet = [0 1 0 0; 0 0 1 0; 0 0 0 1.] .- [1,0,0]
    face = tet[:,[1,2,3]]
    pt = [0,100,0]
    n = cross(face[:,3] - face[:,1],face[:,2] - face[:,1])
    n = n/norm(n)
    pt = pt - dot(pt - face[:,1],n)*n
    @test point_in_polygon(pt,face) == false
  end

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

  @testset "simplex_size" begin
    simplex = [0 0 1; 0 1 0]
    @test simplex_size(simplex) ≈ chull(Array(simplex')).volume

    simplex = [0.0 0.5 0.0; 1.0 0.0 0.0]
    @test simplex_size(simplex) ≈ chull(Array(simplex')).volume
  end
  @testset "carttobary" begin
    simplex = [0 0 1; 0 1 1]
    pt = [0,1]
    carttobary(pt,simplex) ≈ [0,1,0]
    carttobary(simplex,simplex) ≈ [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
  end
end