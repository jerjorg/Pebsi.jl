using Test

import Pebsi.Polynomials: sample_simplex,barytocart,getpoly_coeffs
import Pebsi.QuadraticIntegration: order_vertices!,quadval_vertex, 
    edge_intersects,simplex_intersects,two₋intersects_area₋volume

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

    @testset "two₋intersects_area₋volume" begin
        # Intersections on same edge
        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.89 -0.08 -1.28 1.12 -0.081 -0.88]
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.3533719907367465)
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9039504516362464)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.004871990736746545)
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.09604954836375351)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.18999999999999995 0.1900000000000004 4.1899999999999995 -0.81 1.1900000000000004 0.18999999999999995]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.14657863091452608)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.0186819715046554)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.853421369085474)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.8753486381713221)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 4.359999999999999 0.3600000000000003 0.3599999999999999 1.36 -0.6400000000000001 0.3599999999999999]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.04690964014139585)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.0026642453936121125)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9530903598586041)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.0293309120602787)

        # Intersections at corners
        simplex_pts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
        coeffs = [0, -1.0, 0, 1.0, 1.0, 3.0]
        bezpts = [simplex_pts;coeffs']
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.2175055439664219)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.04376385991606404)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.782494456033578)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.7104305265827225)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 0.19 4.19 -0.81 1.19 0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.22262222446739754)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.03704829657029935)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.7773777755326025)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.8303816299036323)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 4.19 0.19 0.0 1.19 -0.81 0.0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.22262222446739754)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.03704829657029902)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.7773777755326025)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.8303816299036323)
        
        # Intesections on adjacent edges
        coeffs = -[-0.25, -0.25, 3.75, -0.25, 1.75, 1.75]
        bezpts = [simplex_pts;coeffs']
        isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9018252295753189)
        isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.0956051796364183)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.09817477042468103)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.012271846303085152)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2.7600000000000002 0.7599999999999998 2.7600000000000002 0.2599999999999999 0.2599999999999998 -0.23999999999999994]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.0368045419018124)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.0032374347072699453)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9631954580981875)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.0965707680406032)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2.5 -2.5 -3.5 0.5 -2.5 0.5]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.7063128982348126)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.0366388152913721)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.2936871017651873)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.20330548195803894)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2.1 0.16 2.1 -1.8 -1.8 -3.8]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.5905905561646149)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.8189583087823296)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.409409443835385)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.3122916421156629)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -2.9 -4.9 -2.9 -0.9 -0.9 3.0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.8304753037713659)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.7468342379968316)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.1695246962286342)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.1635009046634983)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 3.56 -0.43 -0.43 0.56 -1.44 -0.43]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.519636530743037)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.23864053012229272)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.48036346925696305)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.47030719678895944)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.4 -0.4 3.56 -1.44 0.5 -0.4]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.5177939586360898)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.23233006399160971)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.4822060413639103)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.4689967306582765)

        # Intersection through one corner
        coeffs = [0.25, 0.25, 0, 0.25, -1.75, -1.75]
        bezpts = [simplex_pts;coeffs']
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.7728223042192478)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.4867305453059273)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.2271776957807522)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.028397212201880247)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 11.0 -3.0 -9.0 5.0 -5.0 0.0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.532944336843379)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.759906911671043)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.4670556631566209)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.5932402474827467)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.04 6.04 -3.15 -3.15 -7.35]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.6636465797927428)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-1.6618790753633141)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.3363534202072573)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.5668790759374145)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -2.16 -0.76 0.14 -0.45 1.83]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.6709735097994168)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.4086650827759972)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.32902649020058305)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.17533175032230172)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 3.6 -0.3 -0.36 0.84 -1.16 0.0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.4044138641017234)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.13171032399787191)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.5955861358982766)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.5683769906645386)        

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.55 -2.15 0.0 -0.059 0.34 2.43]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.5633214369508877)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.30638124747125023)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.4366785630491124)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.30821458183459444)

        # One intersection through corner, same edge
        simplex_pts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
        coeffs = [0.6, -1.0, 0, 1.0, 1.0, 3.0]
        bezpts = [simplex_pts;coeffs']
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.1391071149681362)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.021543960069967342)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.8608928850318638)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.788210626736634)

        simplex_pts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
        coeffs = [0, -1.0, 0.6, 1.0, 1.0, 3.0]
        bezpts = [simplex_pts;coeffs']
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.1391071149681362)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.021543960069958863)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.8608928850318638)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.7882106267366255)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 15.8 3.8 1 7.63 -2.35 0.0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.075822952593191)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.029389853208645244)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.924177047406809)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-4.342723186541978)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 15.8 3.8 0.0 7.63 -2.35 1.44]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.0737637461458446)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.02660765734396458)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9262362538541554)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-4.413274324010631)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1 3.8 15.8 -2.3 7.6 0]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.07401391580679513)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.027972810671625898)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9259860841932048)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-4.344639477338292)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 3.8 15.8 -2.3 7.6 1.4]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.0727392387330977)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.025719792742956233)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9272607612669023)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-4.409053126076289)

        # Case where the tangent lines to the curve at the intersections on the
        # triangle are parallel.
        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.96 -1.04 0.96 -0.04 -0.04 0.96]
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.06283185301964875)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.0012803833607820747)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9371681469282042)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.2946137166941154)
    end
end