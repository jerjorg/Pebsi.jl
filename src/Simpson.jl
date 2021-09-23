module Simpson

using LinearAlgebra: dot,inv, norm
using SymmetryReduceBZ.Utilities: remove_duplicates
using Statistics: mean
using ..Geometry: barytocart, carttobary
using ..Polynomials: eval_poly
using ..Defaults: def_atol,def_rtol

eval_1Dquad_basis(t) = [(1 - t)^2, 2*(1 - t)*t, t^2]
# basis_mat = inv(reduce(hcat,[eval_1Dquad_basis(t) for t=[0,1/2,1]])')
basis_mat = [1 0 0; -0.5 2 -0.5; 0 0 1]
get_1Dquad_coeffs(values) = basis_mat*values
evalpoly1D(t,coeffs)=dot(coeffs,eval_1Dquad_basis(t))

@doc """
    bezcurve_intersects(bezcoeffs;rtol,atol)

Determine where a quadratic curve is equal to zero.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.

# Returns
- `solutions::AbstractVector`: the locations between [0,1) where the quadratic
    equals 0.

# Examples
```jldoctest
using Pebsi.Simpson: bezcurve_intersects
coeffs = [0,1,-1]
bezcurve_intersects(coeffs)
# output
[2/3]
```
"""
function bezcurve_intersects(bezcoeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector
    a,b,c = bezcoeffs
    
    quadterm = a - 2*b + c
    linterm = -2*a + 2*b
    
    # Quadratic curve with no intersections
    if !isapprox(quadterm,0,atol=atol)
        maxval = dot([a,b,c],eval_1Dquad_basis((a - b)/(a - 2*b + c)))
        if !isapprox(maxval,0,atol=atol)
            if (maxval > 0 && quadterm > 0) || (maxval < 0 && quadterm < 0)
                return []
            end
        end
    end
    
    # Constant curve with no intersections
    if isapprox(quadterm,0,atol=atol) && isapprox(linterm,0,atol=atol)
        return []
    end
    
    #     # Case 1: [0,0,0]
    #     if isapprox(a,0,atol=0) && isapprox(b,0,atol=0) && isapprox(c,0,atol=0)
    #         return []
    #     end
    
    # Case 5: [a,0,0]
    if isapprox(b,0,atol=atol) && isapprox(c,0,atol=atol)
        # Intersections at t = [1,1] are excluded.
        return []
    end
    
    # Case 3: [0,b,0]
    if isapprox(a,0,atol=0) && isapprox(c,0,atol=0)
        # intersections at t = (0,1). 
        return [0]
    end
    
    # Case 2: [0,0,c]
    if isapprox(a,0,atol=0) && isapprox(b,0,atol=0)
        # intersections at t = (0,0).
        return []
    end
        
    # Case 7: [a,b,0]
    if isapprox(c,0,atol=atol)
        if isapprox(a,2b,atol=atol)
            # Solution at t = 1 excluded.
            return []
        else
            solutions = [a/(a-2*b)]
            solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
                && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
            return solutions
        end
    end
    
    # Case 4: [0,b,c]
    if isapprox(a,0,atol=0)
        if isapprox(2*b,c,atol=atol)
            return [0]
        else
            solutions = [0,2*b/(2*b-c)]
            solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
                && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
            return solutions
        end
    end
    
    # Case 6: [a,0,c]
    if isapprox(b,0,atol=atol)
        if (a < 0 && c < 0) || (a > 0 && c > 0)
            return []
        elseif isapprox(a+c,0,atol=atol)
            solutions = [0.5]
            return solutions
        else
            solutions = real.([(a - im*sqrt(complex(a))*sqrt(complex(c)))/(a + c), 
                (a + im*sqrt(complex(a))*sqrt(complex(c)))/(a + c)])
            solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
                && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
            return sort(solutions)
        end
    end
    
    # Case 8
    if isapprox(a - 2*b + c,0,atol=atol)
        solutions = [a/(2*(a-b))]
        solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
            && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
        return solutions
    elseif isapprox(b^2 - a*c,0,atol=atol)
        solutions = []
        return solutions
    elseif !isapprox(b^2 - a*c,0,atol=atol) && b^2 - a*c < 0
        return []
    else 
        solutions = [(a - b - sqrt(b^2 - a*c))/(a - 2*b + c), (a - b + sqrt(b^2 - a*c))/(a - 2*b + c)]
        solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
            && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
        return sort(solutions)
    end
    solutions
    end


@doc """
    getdomain(bezcoeffs;atol)

Calculate the interval(s) of a quadratic where it is less than 0 between (0,1).

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `atol::Real`: an absolute tolerance for floating point comparisons.

# Returns
- `reg::AbstractVector`: the region where the quadratic is less than 0.

# Examples
```jldoctest
using Pebsi.Simpson: getdomain
coeffs = [0,1,-1]
getdomain(coeffs)
# output
[2/3,1]
```
"""
function getdomain(bezcoeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector
    vals = [evalpoly1D(t,bezcoeffs) for t=[0,1/2,1]]
    if all(vals .< 0) && !any(isapprox.(vals,0,atol=atol))
        return [0,1]
    elseif all(vals .> 0) && !any(isapprox.(vals,0,atol=atol))
        return []
    end
    intersects = bezcurve_intersects(bezcoeffs)
    shaded = [[0,1]; intersects] |> remove_duplicates |> sort
    
    reg = []
    for i = 1:length(shaded) - 1
        test_pts = collect(range(shaded[i],shaded[i+1],step=(shaded[i+1]-shaded[i])/10))
        test_vals = [evalpoly1D(t,bezcoeffs) for t=test_pts]
        if all(x -> x < 0 || isapprox(x,0,atol=atol),test_vals)
            reg = [reg; shaded[[i,i+1]]]
        end
    end
    if reg != []
        reg = reg |> remove_duplicates |> sort
    end
    
    if length(reg) == 3
        if isapprox(sum(diff(reg)),1,atol=atol)
            reg = [0,1]
        end
    end
    reg
end

@doc """
    analytic_area(coeffs,limits)

Calculate the area of a quadratic where it is less than zero between (0,1).

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `limits::AbstractVector`: the interval(s) where the quadratic is less 
    than zero.

# Returns
- `area::Real`: the area under the quadratic

# Examples
```jldoctest
using Pebsi.Simpson: analytic_area
coeffs = [0,1,-1]
limits = [0,1]
analytic_area(coeffs,limits)
# output
-0.1481481481481482
```
"""
function analytic_area(coeffs::AbstractVector{<:Real},limits::AbstractVector)::Real
    if length(limits) == 0
        area = 0
    elseif length(limits) == 2
        a,b,c = coeffs
        t0,t1 = limits
        area = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
          3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
    elseif length(limits) == 4
        a,b,c = coeffs
        t0,t1 = limits[1:2]
        area1 = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
                  3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
        t0,t1 = limits[3:4]
        area2 = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
                  3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
        area = area1 + area2
    else
        error("More limits than expected in the 1D quadratic area calculation.")
    end
    area
end

@doc """
    simpson(y,int_len)

Integrate the area below a list of values within an interval.

# Arguments
- `y::AbstractVector{<:Real}`: a list of values of the function being integrated.
- `int_len::Real`: the length of the interval over which the funcion is integrated.

# Examples
```jldoctest
using Pebsi.Simpson: simpson
f(x)=x^3+x^2+1
v=map(x->f(x),range(-1,3,step=0.1))
simpson(v,4)
# output
33.333333333333336
``` 
"""
function simpson(y::AbstractVector{<:Real},int_len::Real)::Real
    n = length(y)-1
    n % 2 == 0 || error("The number of intervals must be odd for Simpson's method.")
    int_len/(3n) * sum(y[1:2:n] + 4*y[2:2:n] + y[3:2:n+1])
end

function simpson2D(coeffs,triangle,n,q=0;values=false)::Real
    
    lengths = [norm(triangle[:,mod1(i,3)] - triangle[:,mod1(i+1,3)]) for i=1:3]
    corner_midpoint_lens = [norm([mean(triangle[:,[mod1(i,3),mod1(i+1,3)]],dims=2)...] - triangle[:,mod1(i+2,3)]) for i=1:3]
    
    edge_ind = findmax(lengths)[2]
    if edge_ind == 1
       order = [2,3,1]  
    elseif edge_ind == 2
        order = [1,2,3]
    else
        order = [3,1,2]
    end

    m = if iseven(n) n + 1 else n end
    dt = 1/(m-1)
    it = range(0,1,step=dt)
    integral_vals = zeros(length(it))
    for (i,t) in enumerate(it)
        bpt = [t,(1-t)/2,(1-t)/2][order]
        e1bpt = [t,0,1-t][order]
        e2bpt = [t,1-t,0][order]

        bpts = [e1bpt bpt e2bpt]
        pts = barytocart(bpts,triangle)
        vals = eval_poly(bpts,coeffs,2,2)
        bezcoeffs = get_1Dquad_coeffs(vals)
        domain = getdomain(bezcoeffs)

        if q == 0
            if domain == []
                continue
            elseif length(domain) == 2
                integral_vals[i] = (domain[2] - domain[1])*norm(pts[:,1] - pts[:,end])
            elseif length(domain) == 4
                integral_vals[i] = (domain[2] - domain[1] + domain[4] - domain[3])*norm(pts[:,1] - pts[:,end])
            else
                error("Error computing the integration domain in 1D.")
            end
        elseif q == 1
            if domain == []
                continue
            else
                integral_vals[i] = analytic_area(bezcoeffs,domain)*norm(pts[:,1] - pts[:,end])
            end
        else
            error("Invalid value for `q`.")
        end
    end
    
    if values
        return integral_vals
    end

    edge = triangle[:,[edge_ind,mod1(edge_ind+1,3)]]
    opp_corner = triangle[:,mod1(edge_ind+2,3)]
    simpson(integral_vals,linept_dist(edge,opp_corner))
end

@doc """
    linept_dist(line,pt)

Calculate the shortest distance between a point and a line embedded in 2D.

# Arguments
- `line::Matrix{<:Real}`: the endpoints of a line segment as columns of an matrix.
- `pt::Vector{<:Real}`: the coordinates of a point in a vector

# Example
```jldoctest
using Pebsi.Simpson: linept_dist
line = [0 1; 0 0]
pt = [0,2]
# output
2.0
"""
function linept_dist(line,pt)::Real
    unit_vec = [0 -1; 1 0]*(line[:,2] - line[:,1])/norm(line[:,2] - line[:,1])
    abs(dot(unit_vec,pt-line[:,1]))
end

end # module