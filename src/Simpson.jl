module Simpson

using LinearAlgebra: dot,inv,norm,cross
using Statistics: mean
using ..Geometry: barytocart, carttobary, sample_simplex
using ..Polynomials: eval_poly, getpoly_coeffs, eval_1Dquad_basis, 
    get_1Dquad_coeffs, evalpoly1D
using ..Defaults: def_atol,def_rtol
using ..QuadraticIntegration: bezcurve_intersects, getdomain, quad_area₋volume

export bezcurve_intersects, getdomain, analytic_area1D, simpson, simpson2D, 
    linept_dist, tetface_areas, simpson3D

@doc """
    analytic_area1D(coeffs,limits)

Calculate the area of a quadratic where it is less than zero between (0,1).

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `limits::AbstractVector`: the interval(s) where the quadratic is less 
    than zero.

# Returns
- `area::Real`: the area under the quadratic

# Examples
```jldoctest
using Pebsi.Simpson: analytic_area1D
coeffs = [0,1,-1]
limits = [0,1]
analytic_area1D(coeffs,limits)
# output
-0.1481481481481482
```
"""
function analytic_area1D(coeffs::AbstractVector{<:Real},limits::AbstractVector)::Real
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
                integral_vals[i] = analytic_area1D(bezcoeffs,domain)*norm(pts[:,1] - pts[:,end])
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

face_ind = [[1,2,3],[2,3,4],[3,4,1],[4,1,2]]
corner_ind = [4,1,2,3]
function tetface_areas(tet)
    areas = zeros(4)
    for (i,j)=enumerate(face_ind)
        areas[i] = norm(cross(tet[:,j[2]] - tet[:,j[1]], tet[:,j[3]] - tet[:,j[1]]))/2
    end
    areas
end

@doc """
    simpson3D(coeffs,tetrahedron,num_slices,quantity;values)

Calculate the volume or hypervolume beneath a quadratic within a tetrahedron.

# Arguments
- `coeffs`: the coefficients of the quadratic polynomial over the tetrahedron.
- `tetrahedron`: the Cartesian coordinates of the point at the corner of the 
    tetrahedron.
- `num_slices`: the number of slices of teterahedron parallel to one of the 
    faces of the tetrahedron.
- `quantity`: whether to calculate the "area" or "volume" of each slice.
- `values`: if true, return the areas or volumes of each of the slices.

# Returns
- The areas or volumes of slices of the tetrahedron or the volume or hypervolume
    of a polynomial within the tetrahedron.
"""
function simpson3D(coeffs,tetrahedron,num_slices,quantity;values=false)
    dim = 3; deg = 2
    # Area of faces
    face_areas = tetface_areas(tetrahedron)
    p = findmax(face_areas)[2]

    if p == 1
        order = [2,3,4,1]
    elseif p == 2
        order = [1,2,3,4]
    elseif p == 3
        order = [2,1,3,4]
    else
        order = [2,3,1,4]
    end
    
    m = if iseven(num_slices) num_slices + 1 else num_slices end
    dt = 1/(m-1)
    it = range(0,1,step=dt)
    integral_vals = zeros(length(it))
    for (i,t) in enumerate(it)

        bpts = [[t,(1-t),0,0][order],
            [t,(1-t)/2,(1-t)/2,0][order],
            [t,0,(1-t),0][order],
            [t,(1-t)/2,0,(1-t)/2][order],
            [t,0,(1-t)/2,(1-t)/2][order],
            [t,0,0,1-t][order]]
        bpts = reduce(hcat,bpts)
        pts = barytocart(bpts,tetrahedron)
        vals = eval_poly(bpts,coeffs,dim,deg)
        bpts2D = sample_simplex(2,2)
        coeffs2D = getpoly_coeffs(vals,bpts2D,2,2)
        bezpts2D = [pts; coeffs2D']
        integral_vals[i] = quad_area₋volume(bezpts2D,quantity)
    end

    if values
        return integral_vals
    end

    # Calculate the shortest distance from the corner to the opposite face of the tetrahedron.
    face = tetrahedron[:,face_ind[p]]
    corner = tetrahedron[:,corner_ind[p]]
    n = cross(face[:,2] - face[:,1],face[:,3]-face[:,1])
    n = n ./ norm(n)
    d = abs(dot(corner - face[:,1],n))

    simpson(integral_vals,d)
end

end # module