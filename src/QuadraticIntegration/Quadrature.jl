@doc """
    getdomain(bezcoeffs;atol)

Calculate the interval(s) of a quadratic where it is less than 0 between (0,1).

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `reg::AbstractVector`: the region(s) where the quadratic is less than 0.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: getdomain
coeffs = [0,1,-1]
getdomain(coeffs)
# output
2-element Vector{Any}:
 0.6666666666666666
 1.0
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
    analytic_area1D(coeffs,limits)

Calculate the area of a quadratic where it is less than zero between [0,1].

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `limits::AbstractVector`: the interval(s) where the quadratic is less 
    than zero.

# Returns
- `area::Real`: the area under the quadratic

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: analytic_area1D
coeffs = [0.,1.,-1.]
limits = [0.,1.]
analytic_area1D(coeffs,limits)
# output
0.0
```
"""
function analytic_area1D(coeffs::AbstractVector{<:Real}, limits::AbstractVector)::Real
    if length(limits) == 0
        area = 0.
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

Integrate the area below a curve with Simpson's method.

# Arguments
- `y::AbstractVector{<:Real}`: a list of values of the curve being integrated.
- `int_len::Real`: the length of the interval over which the curve is integrated.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simpson
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

@doc """
    linept_dist(line,pt)

Calculate the shortest distance between a point and a line embedded in 2D.

# Arguments
- `line::Matrix{<:Real}`: the endpoints of a line segment as columns of an matrix.
- `pt::Vector{<:Real}`: the coordinates of a point in a vector

# Example
```jldoctest
using Pebsi.QuadraticIntegration: linept_dist
line = [0 1; 0 0]
pt = [0,2]
linept_dist(line,pt)
# output
2.0
```
"""
function linept_dist(line,pt)::Real
    unit_vec = [0 -1; 1 0]*(line[:,2] - line[:,1])/norm(line[:,2] - line[:,1])
    abs(dot(unit_vec,pt-line[:,1]))
end

face_ind = [[2,3,4],[1,3,4],[1,4,2],[1,2,3]]
corner_ind = [1,2,3,4]

# Labeled by corner opposite the face
# The order of the sample points of a slice of the tetrahedron
slice_order1 = [4,2,3,1]
slice_order2 = [1,4,2,3]
slice_order3 = [1,3,4,2]
slice_order4 = [1,2,3,4]

# The order of the coefficients of the 3D quadratic polynomial when the slices
# area towards different corners
coeff_order1 = [3, 8, 10, 5, 9, 6, 2, 7, 4, 1]
coeff_order2 = [1, 4, 6, 7, 9, 10, 2, 5, 8, 3]
coeff_order3 = [1, 7, 10, 2, 8, 3, 4, 9, 5, 6]
coeff_order4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# The order of the vertices of the tetrahedron when the slices are taken towards
# different corners of the tetrahedron
vert_order1 = [4,1,3,2]
vert_order2 = [1,4,2,3]
vert_order3 = [1,3,4,2]
vert_order4 = [1,2,3,4]

@doc """
    tetface_areas(tet)

Calculate the area of the faces of a tetrahedron

# Arguments
- `tet::AbstractMatrix{<:Real}`: the points at the corners of the tetrahedron as
    the columns of a matrix.

# Returns
- `areas::Vector{<:Real}`: the areas of the faces of the tetrahedron. The order
    of the faces is determined by `face_ind`.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: tetface_areas
tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
tetface_areas(tet)
# output
4-element Vector{Float64}:
 0.8660254037844386
 0.5
 0.5
 0.5
```
"""
function tetface_areas(tet::AbstractMatrix{<:Real})::Vector{<:Real}
    areas = zeros(4)
    for (i,j)=enumerate(face_ind)
        areas[i] = norm(cross(tet[:,j[2]] - tet[:,j[1]], tet[:,j[3]] - tet[:,j[1]]))/2
    end
    areas
end

@doc """
    simpson3D(bezpts,quantity;num_slices,values,gauss,split,corner,atol,rtol)

Calculate the volume or hypervolume beneath a quadratic within a tetrahedron.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the quadratic polynomial over a 
    tetrahedron.
- `num_slices::Integer`: the number of slices of teterahedron parallel to one of the faces 
    of the tetrahedron.
- `quantity::String`: whether to calculate the "area" or "volume" of each slice.
- `values::Bool`: if true, return the areas or volumes of each of the slices.
- `gauss::Bool=true`: using Gaussian quadrature points if true.
- `atol::Real=def_atol`: an absolute tolerance.
- `split::Bool=true`: if true, split the integration interval where the slices are
    tangent to the quadratic surface.
- `corner::Union{Nothing,Integer}`: if provided, sliced approach the provided corner. 

# Returns
- The areas or volumes of slices of the tetrahedron or the volume or hypervolume
  of a polynomial within the tetrahedron. May instead be the values of curve being 
  integrated if `values` is true.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simpson3D
spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0; 
    0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0]
coeffs = [-1/10, -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
bezpts= [spts; coeffs']
simpson3D(bezpts,"area",num_slices=10) ≈ 0.01528831567698499

# output
true
```
"""
function simpson3D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices, 
    values::Bool=false, gauss::Bool=true, split::Bool=true, corner::Union{Nothing,Integer}=nothing,
    atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(tetrahedron)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 3; deg = 2; cind = simplex_cornerpts(dim,deg); tetrahedron = bezpts[1:end-1,cind]
     # Area of faces
    if corner == nothing
        face_areas = tetface_areas(tetrahedron)
        p = findmax(face_areas)[2]
    else
        p = corner
    end 
    # Calculate the shortest distance from a plane at the opposite face to the
    # opposite corner
    face = tetrahedron[:,face_ind[p]]
    corner = tetrahedron[:,corner_ind[p]]
    n = cross(face[:,2] - face[:,1],face[:,3]-face[:,1])
    n = n ./ norm(n)
    d = abs(dot(corner - face[:,1],n))
 
    # Reorder coefficients and vertices
    coeff_order = @eval $(Symbol("coeff_order"*string(p)))
    slice_order = @eval $(Symbol("slice_order"*string(p)))
    vert_order = @eval $(Symbol("vert_order"*string(p)))

    if !split
        intervals = [0,1]
    else
        tpts = quadslice_tanpt(coeffs[coeff_order])[vert_order,:]
        if insimplex(tpts[:,1],atol=atol)
            if insimplex(tpts[:,2],atol=atol)
                intervals = [0; tpts[p,:]; 1]
            else
                intervals = [0,tpts[p,1],1]
            end
        elseif insimplex(tpts[:,2],atol=atol) 
            intervals = [0,tpts[p,2],1]
        else
            intervals = [0,1]
        end
    end
     
    interval_lengths = d*diff(intervals)
    interval_divs = [x < 3 ? 3 : mod(x,2) == 0 ? x+1 : x for x=round.(Int,num_slices*interval_lengths)]
    integral = 0; integral_vals = []; its = []
    bpts2D = sample_simplex(2,2)
    for j = 1:length(interval_divs)
        if gauss
            x,w = gausslegendre(interval_divs[j])
            w = w ./ 2
            it = (intervals[j+1] - intervals[j])*(x ./ 2) .+ (intervals[j] + intervals[j+1])/2
        else
            it = range(intervals[j],stop=intervals[j+1],length=interval_divs[j])
        end
        intvals = zeros(length(it))

        # No need to consider the end points; they are always zero
        for (i,t) in enumerate(it)
            bpts = reduce(hcat,[[(1-t),0,0,t],
            [(1-t)/2,(1-t)/2,0,t],
            [0,(1-t),0,t],
            [(1-t)/2,0,(1-t)/2,t],
            [0,(1-t)/2,(1-t)/2,t],
            [0,0,1-t,t]])
            bpts = bpts[slice_order,:]
            pts = barytocart(bpts,tetrahedron)
            vals = eval_poly(bpts,coeffs,dim,deg)
            coeffs2D = getpoly_coeffs(vals,bpts2D,2,2)
            intvals[i] = quad_area_volume([pts; coeffs2D'],quantity)
        end
        if intvals[end] === NaN
            intvals[end] = 0
        end

        if values
            integral_vals = [integral_vals; intvals]
            its = [its; it]
            continue
        end

        if gauss
            integral += interval_lengths[j]*dot(w,intvals)
        else
            integral += simpson(intvals,interval_lengths[j])
        end
    end

    if values
        its,integral_vals
    else
        integral
    end
end

@doc """
    quadslice_tanpt(coeffs,atol,rtol)

Calculate the points where a quadratic surface in tangent to a slice of a tetrahedron.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `atol::Real=def_atol`: an absolute tolerance.
- `rtol::Real=def_rtol`: a relative tolerance.

# Returns
- `bpts::Matrix{Real}`: the points where slices of a tetrahedron parallel to a face
    of the tetrahedron are tangent to the quadratic. The points are in Barycentric
    coordinates.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: quadslice_tanpt
coeffs = [-1/10., -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
quadslice_tanpt(coeffs)
# output
4×2 Matrix{Float64}:
  1.31623   0.683772
  0.0       0.0
  0.0       0.0
 -0.316228  0.316228
```
"""
function quadslice_tanpt(coeffs::AbstractVector{<:Real}; atol::Real=def_atol,
    rtol::Real=def_rtol)
     
    # s^2, 2 s t, t^2, 2 s u, 2 t u, u^2, 2 s v, 2 t v, 2 u v, v^2
    (c2000,c1100,c0200,c1010,c0110,c0020,c1001,c0101,c0011,c0002) = coeffs

    sa = ((c0110-c0200-c1010+c1100)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    sb = -((2*(c0110-c0200-c1010+c1100)*(c0002*c0110^2+
        c0101*c0110*c1001-c0110^2*c1001-c0101^2*c1010-
        c0002*c0110*c1010+c0101*c0110*c1010+c0002*c0200*c1010+
        c0011^2*(c0200-c1100)-c0002*c0110*c1100+
        c0020*(c0101^2+c0200*c1001+c0002*(-c0200+c1100)-
        c0101*(c1001+c1100))+c0011*(-c0200*(c1001+c1010)+c0110*(c1001+c1100)+
        c0101*(-2*c0110+c1010+c1100)))*(c0110^2+(c1010-
        c1100)^2+c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))))

    sc = (c0110-c0200-c1010+c1100)*(c0002*c0110^4+2*c0110^3*c1001^2-
        c0110^2*c0200*c1001^2-2*c0002*c0110^3*c1010+
        2*c0002*c0110^2*c0200*c1010-4*c0101*c0110^2*c1001*c1010+
        2*c0101*c0110*c0200*c1001*c1010+2*c0101^2*c0110*c1010^2+
        c0002*c0110^2*c1010^2-c0101^2*c0200*c1010^2-
        2*c0002*c0110*c0200*c1010^2+c0002*c0200^2*c1010^2-
        2*c0002*c0110*(c0110^2-c0110*c1010+c0200*c1010)*c1100+
        c0002*c0110^2*c1100^2+c0020^2*(c0200*c1001^2-c0101^2*(c0200-2*c1100)+
        c0002*(c0200-c1100)^2-2*c0101*c1001*c1100)+
        c0011^2*(c0110^2*c0200+c0200*c1010*(2*c0200+c1010-2*c1100)+
        2*c0110*(c1100^2-c0200*(c1010+c1100)))+
        c0020*(-c0110^2*c1001^2-c0200*(c0011^2*c0200+(2*c0110-c0200)*c1001^2+
        2*c0011*c1001*c1010)+2*c0011*(c0011*c0200+c0110*c1001)*c1100-c0011^2*c1100^2+
        c0101^2*(c0110^2+2*c0200*c1010+c1100*(-2*c1010+c1100)-
        2*c0110*(c1010+c1100))-2*c0002*(c0200-c1100)*(c0110^2+c0200*c1010-
        c0110*(c1010+c1100))+2*c0101*(c0011*c0110*(c0200-2*c1100)-c0200*c1001*c1100+
        c0011*c1010*c1100+c0110*c1001*(c1010+2*c1100)))-
        2*c0011*((2*c0110-c0200)*c1001*(-c0200*c1010+c0110*c1100)+
        c0101*(c0110^3-c0200*c1010*c1100-2*c0110^2*(c1010+c1100)+
        c0110*(2*c0200*c1010+c1010^2+c1100^2))))

    ta = ((c0110-c1010-c1100+c2000)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+
        2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    tb = -((2*(c0110-c1010-c1100+c2000)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(-c0110*c1001^2-
        c0002*c0110*c1010+c0101*c1001*c1010+c0110*c1001*c1010+
        c0002*c1010^2-c0101*c1010^2-c0002*c1010*c1100+
        c0002*c0110*c2000+c0011^2*(-c1100+c2000)+
        c0020*(-c1001*(c0101-c1001+c1100)+c0002*(c1100-c2000)+
        c0101*c2000)+c0011*(c0110*c1001+c0101*c1010-2*c1001*c1010+
        c1001*c1100+c1010*c1100-(c0101+c0110)*c2000))))

    tc = ((c0110-c1010-c1100+
        c2000)*(c1010*(-2*c0011*c1001*(c0110-c1010)^2+
        c0110^2*(2*c1001^2+c0002*c1010)-
        2*c0110*c1010*(2*c0101*c1001+c0002*(c1010-c1100))+
        c1010*(2*c0101^2*c1010+c0002*(c1010-c1100)^2)+
        4*c0011*(-c0101+c1001)*c1010*c1100+2*c0011^2*c1100^2-
        2*c0011*c1001*c1100^2)+(-c0101^2*c1010^2-
        c0110^2*(c1001^2+2*c0002*c1010)+
        2*c0110*c1010*(c0101*c1001+c0002*c1010-c0002*c1100)+
        c0011^2*((c0110-c1010)^2-2*(c0110+c1010)*c1100)+
        2*c0011*(c0101*c1010*(2*c0110+c1100)+
        c0110*c1001*(-2*c1010+c1100)))*c2000+
        c0110*(2*c0011*(c0011-c0101)+c0002*c0110)*c2000^2+
        c0020^2*(-2*c0101*c1001*c1100+c0002*(c1100-c2000)^2+
        c1001^2*(2*c1100-c2000)+c0101^2*c2000)+
        c0020*(-2*c0110*c1001^2*c1010+c1001^2*c1010^2+
        2*c0011*c0110*c1001*c1100-2*c0110*c1001^2*c1100-
        2*c0002*c0110*c1010*c1100-4*c0011*c1001*c1010*c1100-
        2*c1001^2*c1010*c1100+2*c0002*c1010^2*c1100-
        c0011^2*c1100^2+c1001^2*c1100^2-2*c0002*c1010*c1100^2+
        2*c0101*c1010*(c0110*c1001+(c0011+2*c1001)*c1100)-
        2*c0101*(c0011*c0110+c1001*c1100)*c2000+
        2*(c0011*c1001*c1010+c0011^2*c1100+
        c0002*c1010*(-c1010+c1100)+c0110*(c1001^2+
        c0002*(c1010+c1100)))*c2000-(c0011^2+2*c0002*c0110)*c2000^2+
        c0101^2*(-c1010^2-2*c1010*c2000+c2000^2))))

    ua = ((c0110^2+(c1010-c1100)^2+c0200*(2*c1010-c2000)-
        2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+
        2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+
        c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-
        c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    ub = -((2*(c0110^2+(c1010-c1100)^2+c0200*(2*c1010-c2000)-
        2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(-c0110*c1001^2+
        c0200*c1001^2+c0002*c0200*c1010-c0200*c1001*c1010-
        c0002*c0110*c1100+c0110*c1001*c1100-c0002*c1010*c1100+
        c0011*(c1001-c1100)*c1100+c0002*c1100^2+
        c0002*(c0110-c0200)*c2000+c0011*c0200*(-c1001+c2000)+
        c0101^2*(-c1010+c2000)+c0101*(c0110*c1001+c1001*c1010+c0011*c1100-
        2*c1001*c1100+c1010*c1100-(c0011+c0110)*c2000))))

    uc = (-2*c0110*c0200*c1001^2*c1010+2*c0200^2*c1001^2*c1010+
        c0002*c0200^2*c1010^2+c0200*c1001^2*c1010^2+
        2*c0110^2*c1001^2*c1100-2*c0110*c0200*c1001^2*c1100-
        2*c0002*c0110*c0200*c1010*c1100-2*c0200*c1001^2*c1010*c1100-
        2*c0002*c0200*c1010^2*c1100+c0002*c0110^2*c1100^2+
        c0200*c1001^2*c1100^2+2*c0002*c0110*c1010*c1100^2+
        2*c0002*c0200*c1010*c1100^2+c0002*c1010^2*c1100^2-
        2*c0002*c0110*c1100^3-2*c0002*c1010*c1100^3+
        c0002*c1100^4-(c0110-c0200)*(-c0200*(c1001^2+2*c0002*c1010)+
        2*c0002*(c1010-c1100)*c1100+
        c0110*(c1001^2+2*c0002*c1100))*c2000+
        c0002*(c0110-c0200)^2*c2000^2-
        2*c0011*c1001*(c0200*c1010-c0110*c1100)*(c0200-2*c1100+
        c2000)+c0011^2*(c0200-2*c1100+c2000)*(-c1100^2+c0200*c2000)+
        c0101^2*(2*c1010^2*c1100-c0200*(c1010-c2000)^2-
        2*c1010*(c0110+c1100)*c2000+
        c2000*((c0110-c1100)^2+2*c0110*c2000))+
        2*c0101*(-c0110^2*c1001*c1100-
        c1001*c1100*((c1010-c1100)^2+c0200*(2*c1010-c2000))+
        c0011*c1010*c1100*(c0200-2*c1100+c2000)+
        c0110*(2*c1001*c1100*(c1100-c2000)+c1001*c1010*c2000+
        c0011*(2*c1100-c2000)*c2000+c0200*(c1001*c1010-c0011*c2000))))

    va = (c0002*c0110^2+2*c0101*c0110*c1001-2*c0110^2*c1001-
        2*c0110*c1001^2+c0200*c1001^2-2*c0101^2*c1010-
        2*c0002*c0110*c1010+2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+2*c0110*c1001*c1100-
        2*c0002*c1010*c1100+2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+
        c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-
        c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000)))

    vb = -((2*(-c0011*c0200*c1010-c0200*c1001*c1010-c0101*c1010^2+
        c0200*c1010^2+c0011*c1010*c1100+c0101*c1010*c1100-
        c0011*c1100^2+c0011*c0200*c2000+c0110^2*(-c1001+c2000)+
        c0020*(-(c0101+c1001-c1100)*c1100+c0200*(c1001-c2000)+
        c0101*c2000)+c0110*(c0101*c1010+c1001*c1010+c0011*c1100+c1001*c1100-
        2*c1010*c1100-(c0011+c0101)*c2000))))

    vc = (-2*c0110*c1010*c1100+c0020*c1100^2+c0110^2*c2000+
        c0200*(c1010^2-c0020*c2000))

    abcs = [[sa,sb,sc],[ta,tb,tc],[ua,ub,uc],[va,vb,vc]]
    stuv = [[0.,0.],[0.,0.],[0.,0.],[0.,0.]]
    for (i,abc) in enumerate(abcs)
        a,b,c=abc
        sol = solve_quadratic(abc[1],abc[2],abc[3],atol=atol)
        if length(sol) == 0
            sol = [0,0]
        elseif length(sol) == 1
            sol = [sol[1],sol[1]]
        end
        stuv[i] = sol
    end
     
    s,t,u,v = stuv
    bpts = zeros(4,2); b1 = zeros(4); b2 = zeros(4)
    for i=1:2, j=1:2, k=1:2, l=1:2
        b1 = [s[l],t[k],u[j],v[i]]
        if isapprox(sum(b1),1, atol=atol,rtol=rtol)
            b2 = [s[mod1(l+1,2)],t[mod1(k+1,2)],u[mod1(j+1,2)],v[mod1(i+1,2)]]
            if isapprox(sum(b2),1,atol=atol,rtol=rtol)
                bpts = [b1 b2]
                break
            end
        end
    end
    bpts
end

@doc """
    length_area1D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the area or length of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the bezier points of the univariate polynomial.
- `quantity::String`: the quantity calculate. "area" gives the length of the domain
    where the polynomial is less than zero, and "volume" gives the area (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the polynomial at the quadrature points when true.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The length of the domain or the area below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
coeffs = [1.1,2.2,3.3,4.4]
spts = [-6/3,-1/3,4/3,9/3]
bezpts = [spts'; coeffs']
num_slices = 10
quantity = "area"
length_area1D(bezpts,quantity,num_slices=num_slices)
# output
0.0
```
"""
function length_area1D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices,
    gauss::Bool=true, values::Bool=false, atol::Real=def_atol)

    interval = [bezpts[1,1],bezpts[1,end]]
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return abs(interval[2] - interval[1])
        elseif quantity == "volume"
            return mean(coeffs)*abs(interval[2] - interval[1])
        end 
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 1; deg = length(coeffs) - 1
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    bpts = reduce(hcat,[[i,1-i] for i=it])
    vals = eval_poly(bpts,coeffs,dim,deg)
    if quantity == "area"
        replace!(x -> (x > 0 || isapprox(x,0,atol=atol)) ? 0 : 1, vals)
    elseif quantity == "volume"
        replace!(x -> (x > 0 || isapprox(x,0,atol=atol)) ? 0 : x, vals)
    else
        error("Invalid quantity")
    end    
    if values
        return it,vals
    end

    if gauss
        abs(interval[2] - interval[1])*dot(w,vals)
    else
        simpson(vals,abs(interval[2]-interval[1]))
    end
end

@doc """
    area_volume2D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the area or length of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the bivariate polynomial over a 
    triangle.
- `quantity::String`: the quantity calculate. "area" gives the area of the domain
    where the polynomial is less than zero, and "volume" gives the volume (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of slices. Each slices has this many quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the integral of the polynomial at the quadrature points when true.
- `edge_ind::Union{Nothing,Integer}=nothing`: if provided, slices of triangle start
    from this edge.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The area of the domain or the volume below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
spts = [0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
coeffs = [0.1, 0.2, 0.4, 0.4, 0.3, 0.3]
bezpts = [spts; coeffs']
area_volume2D(bezpts,"area")
# output
0.0
```
"""
function area_volume2D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices,
    gauss::Bool=true, values::Bool=false, edge_ind::Union{Nothing,Integer}=nothing, atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(triangle)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 2; deg = 0; n = 0 
    while n != length(coeffs)
        deg += 1
        n = ntripts(deg)
    end
    cind = simplex_cornerpts(dim,deg)
    triangle = bezpts[1:end-1,cind]
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    if edge_ind == nothing
        # Move from the edge of the triangle of the largest length to the opposite corner
        lengths = [norm(triangle[:,mod1(i,3)] - triangle[:,mod1(i+1,3)]) for i=1:3]
        corner_midpoint_lens = [norm([mean(triangle[:,[mod1(i,3),mod1(i+1,3)]],dims=2)...] -
                triangle[:,mod1(i+2,3)]) for i=1:3]
        edge_ind = findmax(lengths)[2]
    end
    if edge_ind == 1
       order = [2,3,1]  
    elseif edge_ind == 2
        order = [1,2,3]
    else
        order = [3,1,2]
    end 
    ibpts= sample_simplex(1,deg)
    intvals = zeros(length(it))
    for (i,t) in enumerate(it)
        bpt = [t,(1-t)/2,(1-t)/2][order]
        e1bpt = [t,0,1-t][order]
        e2bpt = [t,1-t,0][order]
        bpts = [e1bpt e2bpt]
        pts = barytocart(bpts,triangle)
        polypts = reduce(hcat,[(1-t)*pts[:,1] + t*pts[:,2] for t=0:1/(deg):1])
        polybpts = carttobary(polypts,triangle)
        vals = eval_poly(polybpts,coeffs,dim,deg)
        l = norm(pts[:,1] - pts[:,end])
        if deg < 3
            bezcoeffs = get_1Dquad_coeffs(vals)
        else
            bezcoeffs = getpoly_coeffs(vals,ibpts,1,deg)
        end
        intvals[i] = length_area1D([barytocart(ibpts,Matrix([0,l]')); bezcoeffs'], quantity, 
            num_slices=num_slices, gauss=gauss, atol=atol)
    end
    
    if values
        return it,intvals
    end
    edge = triangle[:,[edge_ind,mod1(edge_ind+1,3)]]
    opp_corner = triangle[:,mod1(edge_ind+2,3)]
    d = linept_dist(edge,opp_corner)
    if gauss
        d*dot(w,intvals)
    else
        simpson(intvals,d)
    end
end

@doc """
    volume_hypvol3D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the volume or hyper volume of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the trivariate polynomial over a
    tetrahedron.
- `quantity::String`: the quantity calculate. "area" gives the area of the domain
    where the polynomial is less than zero, and "volume" gives the volume (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of slices. Each slices has this many quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the integral of the polynomial at the quadrature points when true.
- `corner::Union{Nothing,Integer}=nothing`: if provided, slices of tetrahedron approach this corner.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The volume of the domain or the hyper volume below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0];
coeffs = collect(1:10)
bezpts = [spts; coeffs']
quantity = "area"
volume_hypvol3D(bezpts,quantity)
# output
0.0
```
"""
function volume_hypvol3D(bezpts::Matrix{<:Real}, quantity::String;
    num_slices::Integer=def_num_slices, gauss::Bool=true, values::Bool=false, 
    corner::Union{Nothing,Integer}=nothing, atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(tetrahedron)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end    
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 3; deg = 0; n = 0
    while n != length(coeffs)
        deg += 1
        n = ntetpts(deg)
    end
    cind = simplex_cornerpts(dim,deg)
    tetrahedron = bezpts[1:end-1,cind] 
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    
    # Area of faces
    if corner == nothing
        face_areas = tetface_areas(tetrahedron)
        p = findmax(face_areas)[2]
    else
        p = corner
    end
   
    # Calculate the shortest distance from a plane at the opposite face to the
    # opposite corner
    face = tetrahedron[:,face_ind[p]]
    corner = tetrahedron[:,corner_ind[p]]
    n = cross(face[:,2] - face[:,1],face[:,3]-face[:,1])
    n = n ./ norm(n)
    d = abs(dot(corner - face[:,1],n))
    
    # Reorder coefficients and vertices
    slice_order = @eval $(Symbol("slice_order"*string(p)))
    intvals = zeros(length(it))
    bpts2D = sample_simplex(2,deg)
    # No need to consider the end points; they are always zero
    for (i,t) in enumerate(it)
        bpts = reduce(hcat,[[(1-t),0,0,t],[0,(1-t),0,t],[0,0,1-t,t]])
        bpts = bpts[slice_order,:]
        pts = barytocart(bpts,tetrahedron)
        polypts = barytocart(bpts2D,pts)
        polybpts = carttobary(polypts,tetrahedron)
        vals = eval_poly(polybpts,coeffs,dim,deg)
        coeffs2D = getpoly_coeffs(vals,bpts2D,2,deg)
        triangle = Utilities.mapto_xyplane(pts)
        intvals[i] = area_volume2D([barytocart(bpts2D,triangle); coeffs2D'],quantity,num_slices=num_slices,
            gauss=gauss,atol=atol)
    end
    if values return it,intvals end
    if gauss
        d*dot(w,intvals)
    else
        simpson(intvals,d)
    end        
end
