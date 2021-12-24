using Test

using Pebsi.Geometry: order_vertices!, insimplex, sample_simplex
using Pebsi.Polynomials: barytocart, getpoly_coeffs
using Pebsi.QuadraticIntegration
using Pebsi.QuadraticIntegration: coeff_order1, coeff_order2, coeff_order3, coeff_order4,
    vert_order1, vert_order2, vert_order3, vert_order4, slice_order1, slice_order2, slice_order3,
    slice_order4

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

        x₀=0 #
        y₀=0
        r=1
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [1.0; 0.0], [0.0; 1.0]])

        x₀=0 #
        y₀=0.5
        r=0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[0.0; 0.0], [0.5; 0.5], [0.0 -0.5; 1.0 0.5]])

        x₀=0.5 #
        y₀=0.5
        r=1
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.3660254037844386; 0.0], [], [-0.5; 0.5]])

        x₀=0 #
        y₀=0.5
        r=0.3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[],[]])

        x₀=0 #
        y₀=0.3
        r=0.3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[],[]])

        x₀=0.5 #
        y₀=0.0
        r=0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[0.0; 0.0], [1.0 0.5; 0.0 0.5], []])

        x₀=0.0 #
        y₀=1/3
        r=2/3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.5773502691896257 0.5773502691896253; 0.0 0.0], [0.6666666666666667; 0.3333333333333333], [-5.551115123125783e-17 -0.6666666666666667; 1.0 0.33333333333333326]])

        x₀=0 #
        y₀=-1
        r=sqrt(2)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [1.0; 0.0], []])

        x₀=0.5 #
        y₀=-1
        r=sqrt(13/4)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.9999999999999998; 0.0], [0.21922359359558496; 0.780776406404415], [-0.5000000000000004; 0.49999999999999956]])

        x₀=-0.5 #
        y₀=0.5
        r=sqrt(2)/2
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0 0.0; 0.0 0.0], [0.0; 1.0]])

        x₀=-0.5 #
        y₀=-0.5
        r=1/sqrt(2)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.9999999999999999 -1.1102230246251565e-16; 0.0 0.0], [], []])

        x₀=0 #
        y₀=0.5
        r=sqrt(5/4)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-1.0; 0.0], [1.0; -1.110223024625156e-16], Any[]])

        x₀=0 #
        y₀=0.2
        r=0.4
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[-0.34641016151377546 0.34641016151377535; 0.0 0.0], [], []])

        x₀=0 #
        y₀=3/4
        r=1/4
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[], [0.25; 0.75], [0.0 -0.25; 1.0 0.75]])

        x₀=0 #
        y₀=1/2
        r=sqrt(1/8)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects,[[],[], [0.25; 0.75], [-0.25; 0.75]])

        x₀=1 #
        y₀=-2
        r=sqrt(8)
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-1.0000000000000004; 0.0], [0.26794919243112236; 0.7320508075688776], []])

        x₀=0 #
        y₀=0.4142135623730951
        r = 0.4142135623730949
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[0.0; 0.0], [0.2928932188134524; 0.7071067811865476], [-0.2928932188134524; 0.7071067811865476]])

        x₀=0.7 #
        y₀=0.1
        r = sqrt(10)/10
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[0.39999999999999947; 0.0], [1.0 0.6; 2.7755575615628914e-17 0.4], []])

        x₀=0.1 #
        y₀=0.0
        r = sqrt(2)*9/20
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.5363961030678926 0.7363961030678923; 0.0 0.0], [0.5499999999999999; 0.45000000000000007], []])

        x₀=-1/8 #
        y₀=-1/8
        r = 5/(4*sqrt(2))
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.9999999999999999 0.75; 0.0 0.0], [0.5; 0.5], [-0.25000000000000006; 0.75]])

        x₀=-1/3 #
        y₀=1/3
        r = sqrt(5)/3
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-1.0 0.3333333333333335; 0.0 0.0], [0.3333333333333337; 0.6666666666666663], [0.0; 1.0]])

        x₀=-1/4 #
        y₀=1/4
        r = sqrt(2)/2
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.9114378277661477 0.4114378277661477; 0.0 0.0], [0.25; 0.75], [-0.06698729810778058 -0.9330127018922192; 0.9330127018922194 0.06698729810778081]])
            
        x₀=1/8 #
        y₀=1/3
        r = 0.5
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        intersects = simplex_intersects(bezpts)
        @test containsall(intersects, [[-0.2476779962499649 0.4976779962499649; 0.0 0.0], [0.6230981690549108 0.16856849761175607; 0.3769018309450892 0.8314315023882439], []])
            
        x₀=0 #
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
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.0012566370614359185)
        bezpts[end,:] *= -1
        @test isapprox(two₋intersects_area₋volume(bezpts,"area"),0.9371681469282042)
        @test isapprox(two₋intersects_area₋volume(bezpts,"volume"),-0.29458997039476925)
    end
    @testset "quad_area₋volume" begin
        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.0 -0.0 -1.0 1.0 -0.0 1.0]
        isapprox(quad_area₋volume(bezpts,"area"),0.7499999999999998)
        isapprox(quad_area₋volume(bezpts,"volume"),-0.41666666666666663)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),0.24999999999999994)
        isapprox(quad_area₋volume(bezpts,"volume"),-0.08333333333333331)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 0.0 0.0 1.0 1.0 2.0]
        isapprox(quad_area₋volume(bezpts,"area"),0)
        isapprox(quad_area₋volume(bezpts,"volume"),0)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),1)
        isapprox(quad_area₋volume(bezpts,"volume"),-2/3)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 2.0 3.0 4.0 6.0]
        isapprox(quad_area₋volume(bezpts,"area"),0)
        isapprox(quad_area₋volume(bezpts,"volume"),0)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),1)
        isapprox(quad_area₋volume(bezpts,"volume"),-8/3)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 2.0 -1.0 0.0 -2.0]
        isapprox(quad_area₋volume(bezpts,"area"),1/2)
        isapprox(quad_area₋volume(bezpts,"volume"),-1/3)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),1/2)
        isapprox(quad_area₋volume(bezpts,"volume"),-1/3)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.3599999999999999 -1.6400000000000001 0.3599999999999999 -0.6400000000000001 -0.6400000000000001 0.3599999999999999]
        isapprox(quad_area₋volume(bezpts,"area"),0.9114903966053842)
        isapprox(quad_area₋volume(bezpts,"volume"),-0.31637060523065097)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),0.08850963197457389)
        isapprox(quad_area₋volume(bezpts,"volume"),-0.009703930268055777)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -2.0 0.0 -1.0 -1.0 0.0]
        isapprox(quad_area₋volume(bezpts,"area"),1)
        isapprox(quad_area₋volume(bezpts,"volume"),-2/3)
        bezpts[end,:] *= -1
        isapprox(quad_area₋volume(bezpts,"area"),0)
        isapprox(quad_area₋volume(bezpts,"volume"),0)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.0 -1.0 1.0 -0.5 -0.5 0.0]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6426990805103598)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.09075405187900722)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.35730091893274124)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.09075405129862486)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.5 -1.5 -0.5 -0.5 -1.5 -0.5]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8533057386049668)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.5641127904424919)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.1466942614549789)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.0641127899732446)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.16 -0.8399999999999999 1.16 -0.33999999999999997 -0.33999999999999997 0.16]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.28274333882308134)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.01272345036445063)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.7172566692604262)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.17272344971994935)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.0 -1.0 1.0 -0.2999999999999999 -0.2999999999999999 0.3999999999999999]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.28274333858841894)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.012723450364450618)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.7172566626666133)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.1460567826986915)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2.0 -1.0 0.0 0.5 -0.5 1.0]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.321349540231723)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.0453770259395036)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6786504604528073)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.37871035704648387)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.6666666666666667 -1.3333333333333335 0.6666666666666667 -0.6666666666666667 -0.6666666666666667 1.1102230246251565e-16]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8696051009821929)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.2475507320266624)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.1303949007294842)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.025328509579669683)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -4.440892098500626e-16 -2.0000000000000004 -4.440892098500626e-16 -4.440892098500626e-16 -4.440892098500626e-16 1.9999999999999996]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.5707963265492156)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.237462992842751)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.42920367345078425)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.2374629934329191)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 4.440892098500626e-16 -2.9999999999999996 -1.9999999999999996 -0.49999999999999956 -1.4999999999999993 1.0000000000000004]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.9119635716760932)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-1.0269366061928928)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.08803642601035672)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.026936606970545014)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -1.1102230246251565e-16 -1.0000000000000002 2.0 -1.0000000000000002 -2.220446049250313e-16 -1.1102230246251565e-16]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6426990812411272)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.18150810560091363)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.35730092222660437)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.181508102037155)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.1102230246251565e-16 -0.9999999999999998 2.0 2.220446049250313e-16 1.0 2.0]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.14269908163730394)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.014841437052671925)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8573009210966027)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.6815081033453805)        

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -2.220446049250313e-16 -2.0 -2.220446049250313e-16 -1.5000000000000004 -1.5000000000000004 -1.0000000000000002]
        @test isapprox(quad_area₋volume(bezpts,"area"),1)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-1)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0)
        @test isapprox(quad_area₋volume(bezpts,"volume"),0)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.88 -1.12 0.88 -0.3200000000000002 -0.3200000000000002 0.4800000000000001]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.4043848653548269)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.03512210118696362)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.5956151346451732)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.1151221005920487)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.5 -0.5 1.5 -0.25 -0.25 0.0]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.16067477012758996)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.005672128242437946)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8393252295753191)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.33900545997752773)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.125 -0.875 1.125 -0.37500000000000006 -0.37500000000000006 0.12499999999999997]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.3926990813728042)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.024543692832659402)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6073009255497716)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.14954369200028017)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -1.7763568394002505e-15 -4.000000000000002 -4.000000000000002 -1.7763568394002505e-15 -2.0000000000000018 1.9999999999999982]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8264459089354959)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-1.436551911898073)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.1735540902249046)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.10321858829159386)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.0000000000000002 -1.0 1.0000000000000002 -0.414213562373095 -0.414213562373095 0.1715728752538099]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.5390120840052948)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.04623992698974696)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.460987915547353)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.10343088466229206)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2.7999999999999994 -0.5999999999999999 2.7755575615628914e-17 1.0000000000000004 -0.39999999999999997 1.2]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.14853981633974483)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.009393657483653912)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8514601836602552)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.6760603241503206)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.8050000000000002 -1.3950000000000002 0.40499999999999997 -0.29500000000000026 -0.4950000000000001 0.605]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6361725114941859)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.12882493375126755)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.3638274910863316)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.06715826663757014)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.1102230246251565e-16 -1.75 0.5000000000000001 -0.7499999999999999 -0.4999999999999999 0.5000000000000001]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8792173091576725)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.3506870995897753)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.12078268518481917)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.017353768909409293)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -1.3333333333333335 1.3333333333333333 -1.0 -0.3333333333333336 0.0]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8131374946676863)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.28965755282934735)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.1868625053323137)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.0674353303172949)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.12499999999999989 -1.3750000000000002 1.125 -0.8750000000000002 -0.37500000000000017 0.12499999999999989]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.8243487560323316)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.2572654040822762)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.17565124385255756)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.04893207054480276)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 1.1267361111111112 -1.1232638888888888 0.6267361111111112 -0.33159722222222227 -0.5815972222222223 0.21006944444444453]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.6479211505415876)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.09098009530377613)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.35207884972009956)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.0788273166478721)

        bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.47111111111111104 -1.528888888888889 0.47111111111111104 -0.8622222222222224 -0.8622222222222224 -0.1955555555555556]
        @test isapprox(quad_area₋volume(bezpts,"area"),0.9450230872938824)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.4257654090260549)
        bezpts[end,:] *= -1
        @test isapprox(quad_area₋volume(bezpts,"area"),0.054976904256016014)
        @test isapprox(quad_area₋volume(bezpts,"volume"),-0.007987631230349884)

        # Rectangular hyperbola
        dim=2
        deg=2
        simplex_bpts = sample_simplex(2,2)
        triangle = reduce(hcat,[[-1, 0], [1, 0], [0, 1]])
        order_vertices!(triangle)
        simplex_pts = barytocart(simplex_bpts,triangle)
        
        a = 1
        b = a
        x₀=0
        y₀=1/3
        r = 0.8
        f(x,y) = (x-x₀)^2/a^2-(y-y₀)^2/b^2#-r^2
                x₀=0 #
                y₀=1/3
                r = 0.8
        vals = [f(simplex_pts[:,i]...) for i=1:size(simplex_pts,2)]
        coeffs = getpoly_coeffs(vals,simplex_bpts,dim,deg)
        bezpts = [simplex_pts;coeffs']
        @test isapprox(quad_area₋volume(bezpts,"area"),1/3)
    end
    @testset "quadslice_tanpt" begin
        ans_tpts = [[0.18257418583505536, 0.18257418583505536, 0.18257418583505536],
        [0.31622776601683794, 0.0, 0.0],
        [0.0, 0.31622776601683794, 0.0],
        [0.0, 0.0, 0.31622776601683794]]

        tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        coeffs = [-1/10, -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
        for i=1:4
            coeff_order = @eval $(Symbol("coeff_order"*string(i)))
            vert_order = @eval $(Symbol("vert_order"*string(i)))
            coeffsi = coeffs[coeff_order]
            tbpt = quadslice_tanpt(coeffsi)[vert_order,:]    
            if insimplex(tbpt[:,1]) p = 1 else p = 2 end
            tpt = barytocart(tbpt[:,p],tet)
            @test tpt ≈ ans_tpts[i]
        end

        ans_tpts = [[],
            [0.14429938887061264, 0.2672456551994024, 0.41040275575832436],
            [],
            []]
        tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        coeffs = [-1.7, 1.5, 3.1, -5.5, 2.2, 5.1, 2.7, 3.3, -2.8, -1.6]
        for i=1:4
            coeff_order = @eval $(Symbol("coeff_order"*string(i)))
            vert_order = @eval $(Symbol("vert_order"*string(i)))
            coeffsi = coeffs[coeff_order]
            tbpt = quadslice_tanpt(coeffsi)[vert_order,:]    
            if insimplex(tbpt[:,1]) p = 1 elseif insimplex(tbpt[:,2]) p = 2  else p = 0 end
            if p == 0
                @test ans_tpts[i] == []
            else
                tpt = barytocart(tbpt[:,p],tet)
                @test tpt ≈ ans_tpts[i]
            end
        end

        ans_tpts = [[0.27958451488210706, 0.11099974661264694, 0.33023622672501896],
        [],
        [0.19916508112729647, 0.1610281149843057, 0.3228550582398061],
        []]
    
        tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        coeffs = [-1.61, 1.92, -0.46, -0.8, 1.03, -0.88, 0.96, -1.95, -0.27, 0.56]
        for i=1:4
            coeff_order = @eval $(Symbol("coeff_order"*string(i)))
            vert_order = @eval $(Symbol("vert_order"*string(i)))
            coeffsi = coeffs[coeff_order]
            tbpt = quadslice_tanpt(coeffsi)[vert_order,:]    
            if insimplex(tbpt[:,1]) p = 1 elseif insimplex(tbpt[:,2]) p = 2  else p = 0 end
            if p == 0
                @test ans_tpts[i] == []
            else
                tpt = barytocart(tbpt[:,p],tet)
                @test tpt ≈ ans_tpts[i]
            end
        end

        ans_tpts = [[0.4513621002767775, 0.13141495123059868, 0.12038109860944715],
        [0.248161573423525, 0.25685172007121715, 0.04680496590597898],
        [0.03292232846130888, 0.1627482766685301, 0.45411093098063193],
        [0.14400314379590318, 0.8045052911069799, 0.01569541277902617]]
    
        tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        coeffs = [1.32, -1.78, 1.53, 0.25, -0.25, -0.05, -1.46, 0.87, 2.32, -0.25]
        for i=1:4
            coeff_order = @eval $(Symbol("coeff_order"*string(i)))
            vert_order = @eval $(Symbol("vert_order"*string(i)))
            coeffsi = coeffs[coeff_order]
            tbpt = quadslice_tanpt(coeffsi)[vert_order,:]    
            if insimplex(tbpt[:,1]) p = 1 elseif insimplex(tbpt[:,2]) p = 2  else p = 0 end
            if p == 0
                @test ans_tpts[i] == []
            else
                tpt = barytocart(tbpt[:,p],tet)
                @test tpt ≈ ans_tpts[i]
            end
        end
    end
    @testset "simpson3D" begin
        answers = [[0.016557647109660602, -0.0006623058843864066],
            [0.052787355609319415, -0.035171236804781694], 
            [0.1472825340495098, -0.2040822999010772],
            [0.12564018775172114, -0.1330916637183115],
            [0.03924641746264576, -0.048706217271428785]]
         
        tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0; 
        0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0]
        coeffs1 = [-1/10, -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
        coeffs2 = [-1.7, 1.5, 3.1, -5.5, 2.2, 5.1, 2.7, 3.3, -2.8, -1.6]
        coeffs3 = [3.3, 0.2, -0.1, -1.2, 0.9, -5.9, -3.3, -2.8, -0.5, -2.1]
        coeffs4 = [-2.8, -5.5, -2.1, -0.6, -0.3, -0.4, 3.1, 0.8, 0.4, 0.2]
        coeffs5 = [-8.5, 9.1, 0.2, -1.1, -4.4, 0.0, 2.9, 3.8, 10.4, -3.6]
        allcoeffs = [coeffs1, coeffs2, coeffs3, coeffs4, coeffs5]
        allbezpts = [[spts; coeffs'] for coeffs = allcoeffs]
        
        vals = [[simpson3D(bezpts,q,num_slices=100,split=false,corner=4) for q=["area","volume"]] for bezpts=allbezpts]
        test = [abs.(vals[i] .- answers[i]) for i=1:length(vals)]
        @test maximum(maximum(test)) < 5e-6
         
        vals = [[simpson3D(bezpts,q,num_slices=100,split=true,corner=3) for q=["area","volume"]] for bezpts=allbezpts]
        test = [abs.(vals[i] .- answers[i]) for i=1:length(vals)]
        @test maximum(maximum(test)) < 5e-6
    end
    @testset "length_area1D" begin
        interval = [-1,2]
        spts = [-1, 0.5, 2]
        coeffs = [-1,0.5,1.2] 
        bezpts= [spts'; coeffs']
        l1 = length_area1D(bezpts,"area",num_slices=100001,gauss=true)
        domain = getdomain(coeffs);
        domain = [x*(interval[2] - interval[1]) + interval[1] for x=domain]
        l2 = diff(domain)[1]
        @test isapprox(l1,l2,atol=1e-4)
        a1 = length_area1D(bezpts,"volume",num_slices=100001,gauss=true)
        a2 = analytic_area1D(coeffs,getdomain(coeffs))*(interval[2] - interval[1])
        @test isapprox(a1,a2,atol=1e-4)
    end
    @testset "area_volume2D" begin        
        triangle = [0 1 0; 0 0 1]
        sbpts = sample_simplex(2,2)
        spts = barytocart(sbpts,triangle)
        coeffs = [-0.1, -0.2, -0.4, 0.4, 0.3, -0.3]
        bezpts = [spts; coeffs']
        @test isapprox(abs(area_volume2D(bezpts,"area") - quad_area₋volume(bezpts,"area")),0,atol=1e-4)
        @test isapprox(abs(area_volume2D(bezpts,"volume") - quad_area₋volume(bezpts,"volume")),0,atol=1e-6)
    end
    @testset "volume_hypvol3D" begin
        spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0; 
            0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0]
        tetrahedron = [0 1 0 0; 0 0 1 0; 0 0 0 1]
        coeffs = [-0.2,-0.3,0.2,0.3,0.4,-0.5,-0.6,-0.2,0.1,0.2]
        bezpts = [spts; coeffs']
        v1 = simpson3D(bezpts,"area")
        v2 = volume_hypvol3D(bezpts,"area",num_slices=10)
        @test isapprox(abs(v1-v2),0,atol=1e-2)
        
        h1 = simpson3D(bezpts,"volume")
        h2 = volume_hypvol3D(bezpts,"volume",num_slices=10)
        @test isapprox(abs(h1-h2),0,atol=1e-4)
    end
end