using Test

using Pebsi.Simpson: bezcurve_intersects, getdomain, analytic_area, simpson

@testset "Simpson" begin
  @testset "bezcurve_intersects" begin
    # Case 1: [0,0,0]
    coeffs = [0,0,0]
    @test bezcurve_intersects(coeffs) == []
    
    # Case 2: [0,0,c]
    coeffs = [0,0,0.1]
    @test bezcurve_intersects(coeffs) == []
    
    # Case 3: [0,b,0]
    coeffs = [0,0.1,0]
    @test bezcurve_intersects(coeffs) == []
    
    # Case 4: [0,b,c]
    # 2b = c
    coeffs = [0,1,2]
    @test bezcurve_intersects(coeffs) == []
    
    coeffs = [0,1,-1]
    @test bezcurve_intersects(coeffs) == [2/3]
    
    # Case 5: [a,0,0]
    coeffs = [2,0,0]
    @test bezcurve_intersects(coeffs) == []
    
    # Case 6: [a,0,c]
    # a = -c
    coeffs = [-1,0,1]
    @test bezcurve_intersects(coeffs) == [0.5]
    
    coeffs = [-1,0,2]
    @test bezcurve_intersects(coeffs) == [0.41421356237309515]
    
    # a & c > 0
    coeffs = [1,0,2]
    @test bezcurve_intersects(coeffs) == []
    
    # a & c < 0
    coeffs = [-2,0,-3]
    @test bezcurve_intersects(coeffs) == []
    
    # Case 7: [a,b,0]
    # a = 2b
    coeffs = [2,1,0]
    @test bezcurve_intersects(coeffs) == []
    
    coeffs = [-3,2,0]
    @test bezcurve_intersects(coeffs) == [0.42857142857142855]
    
    # Case 8: {a,b,c}
    # a - 2 b + c = 0
    coeffs = [3,2,1]
    @test bezcurve_intersects(coeffs) == []
    
    coeffs = [-3,2,7]
    @test bezcurve_intersects(coeffs) == [0.3]
    
    coeffs = [2,4,8]
    @test bezcurve_intersects(coeffs) == []
    
    coeffs = [1,-3,9]
    @test bezcurve_intersects(coeffs) == [0.25]
    
    coeffs = [1,-3,3]
    @test bezcurve_intersects(coeffs) ≈ [0.15505102572168222,0.6449489742783179]
  end

  @testset "getdomain" begin
    # Case 1: [0,0,0]
    coeffs = [0,0,0]
    @test getdomain(coeffs) ≈ [0,1]
    
    # Case 2: [0,0,c]
    coeffs = [0,0,0.1]
    @test getdomain(coeffs) == []
    
    coeffs = [0,0,-0.1]
    @test getdomain(coeffs) ≈ [0,1]
    
    # Case 3: [0,b,0]
    coeffs = [0,0.1,0]
    @test getdomain(coeffs) == []
    
    coeffs = [0,-0.1,0]
    @test getdomain(coeffs) ≈ [0,1]
    
    # Case 4: [0,b,c]
    # 2b = c
    coeffs = [0,1,2]
    @test getdomain(coeffs) == []
    
    coeffs = [0,-1,-2]
    @test getdomain(coeffs) ≈ [0,1]
    
    coeffs = [0,1,-1]
    @test getdomain(coeffs) ≈ [2/3,1]
    
    coeffs = [0,-1,1]
    @test getdomain(coeffs) ≈ [0,2/3]
    
    # Case 5: [a,0,0]
    coeffs = [2,0,0]
    @test getdomain(coeffs) == []
    
    coeffs = [-2,0,0]
    @test getdomain(coeffs) ≈ [0,1]
    
    # Case 6: [a,0,c]
    # a = -c
    coeffs = [-1,0,1]
    @test getdomain(coeffs) ≈ [0,0.5]
    
    coeffs = [1,0,-1]
    @test getdomain(coeffs) ≈ [0.5,1]
    
    coeffs = [-1,0,2]
    @test getdomain(coeffs) ≈ [0,0.41421356237309515]
    
    coeffs = [1,0,-2]
    @test getdomain(coeffs) ≈ [0.41421356237309515,1]
    
    # a & c > 0
    coeffs = [1,0,2]
    @test getdomain(coeffs) == []
    
    # a & c < 0
    coeffs = [-1,0,-2]
    @test getdomain(coeffs) ≈ [0,1]
    
    # Case 7: [a,b,0]
    # a = 2b
    coeffs = [2,1,0]
    @test getdomain(coeffs) == []
    
    coeffs = [-2,-1,0]
    @test getdomain(coeffs) ≈ [0,1]
    
    coeffs = [-3,2,0]
    @test getdomain(coeffs) ≈ [0,0.42857142857142855]
    
    coeffs = [3,-2,0]
    @test getdomain(coeffs) ≈ [0.42857142857142855,1]
    
    # Case 8: {a,b,c}
    # a - 2 b + c = 0
    coeffs = [3,2,1]
    @test getdomain(coeffs) == []
    
    coeffs = [-3,-2,-1]
    @test getdomain(coeffs) ≈ [0,1]
    
    coeffs = [-3,2,7]
    @test getdomain(coeffs) ≈ [0,0.3]
    
    coeffs = [3,-2,-7]
    @test getdomain(coeffs) ≈ [0.3,1]
    
    coeffs = [2,4,8]
    @test getdomain(coeffs) == []
    
    coeffs = [1,-3,9]
    @test getdomain(coeffs) == []
    
    coeffs = [-1,3,-9]
    @test getdomain(coeffs) ≈ [0,1]
    
    coeffs = [1,-3,3]
    @test getdomain(coeffs) ≈ [0.15505102572168222,0.6449489742783179]
    
    coeffs = [-1,3,-3]
    @test getdomain(coeffs) ≈ [0,0.15505102572168222,0.6449489742783179,1]    
  end

  @testset "analytic_area" begin
    # Case 1: [0,0,0]
    coeffs = [0,0,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    # Case 2: [0,0,c]
    coeffs = [0,0,0.1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [0,0,-0.1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.03333333333333333
    
    # Case 3: [0,b,0]
    coeffs = [0,0.1,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [0,-0.1,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.03333333333333334
    
    # Case 4: [0,b,c]
    # 2b = c
    coeffs = [0,1,2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [0,-1,-2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -1.0
    
    coeffs = [0,1,-1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -4/27
    
    coeffs = [0,-1,1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -4/27
    
    # Case 5: [a,0,0]
    coeffs = [2,0,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [-2,0,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -2/3
    
    # Case 6: [a,0,c]
    # a = -c
    coeffs = [-1,0,1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -1/4
    
    coeffs = [1,0,-1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -1/4
    
    coeffs = [-1,0,2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.2189514164974602
    
    coeffs = [1,0,-2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.5522847498307935
    
    # a & c > 
    coeffs = [1,0,2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    # a & c < 0
    coeffs = [-1,0,-2]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -1.0
    
    # Case 7: [a,b,0]
    # a = 2b
    coeffs = [2,1,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [-2,-1,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -1.0
    
    coeffs = [-3,2,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.5510204081632653
    
    coeffs = [3,-2,0]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.21768707482993196
    
    # Case 8: {a,b,c}
    # a - 2 b + c = 0
    coeffs = [3,2,1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [-3,-2,-1]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -2.0
    
    coeffs = [-3,2,7]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.45
    
    coeffs = [3,-2,-7]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -2.45
    
    coeffs = [2,4,8]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [1,-3,9]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ 0.0
    
    coeffs = [-1,3,-9]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -2.3333333333333335
    
    coeffs = [1,-3,3]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.19595917942265423
    
    coeffs = [-1,3,-3]
    domain = getdomain(coeffs)
    @test analytic_area(coeffs,domain) ≈ -0.5292925127559875
  end

  @testset "simpson" begin
    f(x)=x^3+x^2+1
    v=map(x->f(x),range(-1,3,step=0.1))
    @test simpson(v,4) ≈ 33.333333333333336
  end

end