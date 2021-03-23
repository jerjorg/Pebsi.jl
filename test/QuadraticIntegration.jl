using Test

import Pebsi.Polynomials: sample_simplex,barytocart,getpoly_coeffs
import Pebsi.QuadraticIntegration: order_vertices!,quadval_vertex, 
    edge_intersects,simplex_intersects,same_edge


function contains_intersect(intersects,int1)
contained = false
if int1 == []
    contained = true
else
    for int2 = intersects
        if int2 == []
            continue
        end
        if size(int2,2) == size(int1,2)
            if int1 ≈ int2
                contained = true
            end
        end
    end
end
contained
end

function containsall(intersects1,intersects2)
all([contains_intersect(intersects1,i) for i=intersects2]) & 
    all([contains_intersect(intersects2,i) for i=intersects1])
end

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
        @test edge_intersects(bezpts) == [1.665334536937679e-14]

        coeffs = [-0.6591549430918953, -0.15915494309189532, 0.3408450569081046]
        bezpts = vcat(cartpts,coeffs')
        @test edge_intersects(bezpts) == [0.6591549430918954]
    end

    @testset "simplex_intersects" begin
        dim=2
        deg=2
        simplex_bpts = sample_simplex(dim,deg)
        triangle = reduce(hcat,[[-1, 0], [1, 0], [0, 1]])
        order_vertices!(triangle)
        simplex_pts = barytocart(simplex_bpts,triangle)

        # Planar surface tests
        # Straight line through two edges of triangle
        coeffs = [-1,0,1,-1,0,-1]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[0.0; 0.0], [0.5; 0.5], []])

        coeffs = [-1,-1,-1,0,0,1]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[], [0.5; 0.5], [-0.5; 0.5]])

        coeffs = [1,0,-1,0,-1,-1]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[0.0; 0.0],[],[-0.5; 0.5]])

        # Straight line along edge of triangle
        coeffs = [0,0,0,1,1,2]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        coeffs = [2,1,0,1,0,0]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        coeffs = [0,1,2,0,1,0]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        # Straight line passes through corner outside triangle
        coeffs = [0,1,2,3,4,6]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        coeffs = [2,1,0,4,3,6]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        coeffs = [2,4,6,1,3,0]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test intersects == [[],[],[]]

        # Straight line passes through corner inside triangle
        coeffs = [0,1,2,-1,0,-2]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [0.5; 0.5], []])

        coeffs = [-2,-1,0,0,1,2]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[], [1.0; 0.0], [-0.5; 0.5]])

        coeffs = [2,0,-2,1,-1,0]
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[0.0; 0.0], [], [0.0; 1.0]])        

        # Elipse tests
        f(x,y) = (x-x₀)^2+(y-y₀)^2-r^2
        x₀=0
        y₀=0
        r=0.8
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.8 0.8; 0.0 0.0],[0.7645751311064592 0.23542486889354075; 0.2354248688935408 0.7645751311064592], [-0.2354248688935408 -0.7645751311064592; 0.7645751311064592 0.23542486889354075]])


        x₀=0
        y₀=0
        r=1
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [1.0; 0.0], [0.0; 1.0]])

        x₀=0
        y₀=0.5
        r=0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[], [0.5; 0.5], [0.0 -0.5; 1.0 0.5]])

        x₀=0.5
        y₀=0.5
        r=1
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.3660254037844386; 0.0], [], [-0.5; 0.5]])

        x₀=0
        y₀=0.5
        r=0.3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[],[]])

        x₀=0
        y₀=0.3
        r=0.3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[],[]])

        x₀=0.5
        y₀=0.0
        r=0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[0.0; 0.0], [1.0 0.5; 0.0 0.5], []])

        x₀=0.0
        y₀=1/3
        r=2/3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.5773502691896257 0.5773502691896253; 0.0 0.0], [0.6666666666666667; 0.3333333333333333], [-5.551115123125783e-17 -0.6666666666666667; 1.0 0.33333333333333326]])

        x₀=0
        y₀=-1
        r=sqrt(2)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0000000000000002; 0.0], [1.0; 0.0], []])

        x₀=0.5
        y₀=-1
        r=sqrt(13/4)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.9999999999999998; 0.0], [0.21922359359558496; 0.780776406404415], [-0.5000000000000004; 0.49999999999999956]])

        x₀=-0.5
        y₀=0.5
        r=sqrt(2)/2
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0 0.0; 0.0 0.0], [0.0; 1.0]])

        x₀=-0.5
        y₀=-0.5
        r=1/sqrt(2)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.9999999999999999 -1.1102230246251565e-16; 0.0 0.0], [], []])

        x₀=0
        y₀=0.5
        r=sqrt(5/4)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [1.0; -1.110223024625156e-16], Any[]])

        x₀=0
        y₀=0.2
        r=0.4
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.34641016151377546 0.34641016151377535; 0.0 0.0], [], []])

        x₀=0
        y₀=3/4
        r=1/4
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[], [0.25; 0.75], [0.0 -0.25; 1.0 0.75]])

        x₀=0
        y₀=1/2
        r=sqrt(1/8)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[], [0.25; 0.75], [-0.25; 0.75]])

        x₀=1
        y₀=-2
        r=sqrt(8)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-1.0000000000000004; 0.0], [0.26794919243112236; 0.7320508075688776], []])

        x₀=0
        y₀=0.4142135623730951
        r = 0.4142135623730949
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[],[],[]])

        x₀=0.7
        y₀=0.1
        r = sqrt(10)/10
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[0.39999999999999947; 0.0], [1.0 0.6; 2.7755575615628914e-17 0.4], []])

        x₀=0.1
        y₀=0.0
        r = sqrt(2)*9/20
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.5363961030678926 0.7363961030678923; 0.0 0.0], [0.5499999999999999; 0.45000000000000007], []])

        x₀=-1/8
        y₀=-1/8
        r = 5/(4*sqrt(2))
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.9999999999999999 0.75; 0.0 0.0], [], [-0.25000000000000006; 0.75]])

        x₀=-1/3
        y₀=1/3
        r = sqrt(5)/3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-1.0 0.3333333333333335; 0.0 0.0], [0.3333333333333337; 0.6666666666666663], [0.0; 1.0]])

        x₀=-1/4
        y₀=1/4
        r = sqrt(2)/2
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.9114378277661477 0.4114378277661477; 0.0 0.0], [0.25; 0.75], [-0.06698729810778058 -0.9330127018922192; 0.9330127018922194 0.06698729810778081]])
            
        x₀=1/8
        y₀=1/3
        r = 0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.2476779962499649 0.4976779962499649; 0.0 0.0], [0.6230981690549108 0.16856849761175607; 0.3769018309450892 0.8314315023882439], []])
            
        x₀=0
        y₀=1/3
        r = 0.8
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.7272474743090477 0.7272474743090478; 0.0 0.0], [0.7903769733600696; 0.20962302663993032], [-0.7903769733600696; 0.20962302663993038]])
    end

    @testset "same_edge" begin
        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.89 -0.08 -1.28 1.12 -0.081 -0.88]
        @test isapprox(same_edge(bezpts,"volume"),-0.3533719907367465)
        @test isapprox(same_edge(bezpts,"area"),0.9039504516362464)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.89 0.08 1.28 -1.12 0.081 0.88]
        @test isapprox(same_edge(bezpts,"volume"),-0.004871990736746545)
        @test isapprox(same_edge(bezpts,"area"),0.09604954836375351)
    end

end