module QuadraticIntegration

include("Polynomials.jl")
include("EPMs.jl")

import SymmetryReduceBZ.Utilities: unique_points

import .Polynomials: sample_simplex,eval_poly,getpoly_coeffs,barytocart,
    carttobary, getbez_pts₋wts,eval_bezcurve,conicsection

import .EPMs: eval_epm, RytoeV

import LinearAlgebra: cross,det,norm,dot
using MiniQhull,Delaunay
import Delaunay: Triangulation
import Statistics: mean
import Base.Iterators: flatten
import SparseArrays: findnz

@doc """
    order_vertices(vertices)

Put the vertices of a triangle (columns of an array) in counterclockwise order.
"""
function order_vertices!(vertices::AbstractMatrix{<:Real})
    for i=1:3
        j = mod1(i+1,3)
        k = mod1(i+2,3)
        (v1,v2,v3) = [vertices[:,l] for l=[i,j,k]]
        if cross(vcat(v2-v1,0),vcat(v3-v1,0))[end] < 0
            v = vertices[:,k]
            vertices[:,k] = vertices[:,j]
            vertices[:,j] = v
        end 
    end
    vertices
end


@doc """
    quadval_vertex(bezcoeffs)

Calculate the value of a quadratic curve at its vertex.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the quadratic polynomial coefficients.
"""
function quadval_vertex(bezcoeffs::AbstractVector{<:Real})
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
end

@doc """
    edge_intersects(bezpts;atol)

Calculate where a quadratic curve is equal to zero within [0,1).

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points (columns of an array).
- `atol::Real=1e-12`: absolute tolerance for comparisons of floating point 
    numbers with zero.

# Returns
- `::AbstractVector{<:Real}`: an array of up to two intersections as real numbers.

# Examples
import Pebsi.QuadraticIntegration: edge_intersects
coeffs = [1,0,-1]
bezpts = vcat(cartpts,coeffs')
edge_intersects(bezpts) == [0.5]
# output
1-element Array{Float64,1}:
 0.5
"""
function edge_intersects(bezpts::AbstractMatrix{<:Real};
    atol::Real=1e-12)::AbstractVector{<:Real}
   
    # Cases where the curve is above zero, below zero, or at zero.
    coeffs = bezpts[end,:]
    if all(isapprox.(coeffs,0,atol=atol))
        return Array{Float64}([])
    elseif all(.!isapprox.(coeffs,0,atol=atol) .& (coeffs .> 0))
        return Array{Float64}([])
    elseif all(.!isapprox.(coeffs,0,atol=atol) .& (coeffs .< 0))
        return Array{Float64}([])
    end
    
    # Put the polynomial in a form where cases are easier to handle:
    # α + βx + γx² == 0
    (a,b,c)=coeffs
    α = a
    β = -2a+2b
    γ = a-2b+c
   
    # Quadratic curve entirely above or below zero.
    v = quadval_vertex(coeffs)
    if (all([γ,v].>0) && all(isapprox.([γ,v],0,atol=atol))) ||
       (all([γ,v].<0) && all(isapprox.([γ,v],0,atol=atol))) && abs(v) != Inf
        return Array{Float64}([])
    end
    #if all([γ,v].>0) || all([γ,v].<0) && abs(v) != Inf
    #     return Array{Float64}([])
    #end

    if isapprox(γ,0,atol=atol) && isapprox(β,0,atol=atol)
        return Array{Float64}([])
    elseif isapprox(γ,0,atol=atol)
        x = [-α/β]
    else
        arg = β^2-4α*γ
        if isapprox(arg,0,atol=atol)
            # There are two solutions at the same point if arg == 0 but we only
            # keep one of them.
            x = [-β/(2γ)]
        elseif arg < 0
            # Ignore solutions with imaginary components.
            return Array{Float64}([])
        else
            x = [(-β-sqrt(arg))/(2γ),(-β+sqrt(arg))/(2γ)]
        end
    end

    # Only keep intersections between [0,1).
    filter(y -> (
        (y>0 || isapprox(y,0,atol=atol)) && 
        (y<1 && !isapprox(y,1,atol=atol))
        ),x) |> sort
end

@doc """
The locations of quadratic Bezier points at the corners of the triangle in
counterclockwise order.
"""
corner_indices = [1,3,6]

@doc """
The locations of quadratic Bezier points along each edge of the triangle in
counterclockwise order.
"""
edge_indices=[[1,2,3],[3,5,6],[6,4,1]]

@doc """
    simplex_intersects(bezpts,atol)

Calculate the location where a level curve of a quadratic surface at z=0 intersects a triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic, Bezier
    surface.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `intersects::Array`: the intersections organized by edge in a 1D array. Each 
    element of the array is a 2D array where the columns are the Cartesian
    coordinates of intersections.

# Examples
```jldoctest
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -1.0 -2.0 1.0 0.0 2.0]
simplex_intersects(bezpts)
# output
3-element Array{Array,1}:
 [-1.0; 0.0]
 [0.5; 0.5]
 Any[]
```
"""
function simplex_intersects(bezpts::AbstractMatrix{<:Real};
    atol::Real=1e-12)::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = edge_intersects(edge_bezpts,atol=atol)
        if edge_ints != []
            intersects[i] = reduce(hcat,[edge_bezpts[1:2,1] .+ 
                i*(edge_bezpts[1:2,end] .- edge_bezpts[1:2,1]) for i=edge_ints])
        end
    end
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects == 1
        Array{Array,1}([[],[],[]])
    else
        intersects
    end
end

@doc """
    saddlepoint(coeffs)

Calculate the saddle point of a quadratic Bezier surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `::AbstractVector{<:Real}`: the coordinates of the saddle point in Barycentric coordinates.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: saddlepoint
coeffs = [0.36, -1.64, 0.36, -0.64, -0.64, 0.36]
saddlepoint(coeffs)
# output
3-element Array{Float64,1}:
 0.5000000000000001
 4.163336342344338e-17
 0.5000000000000001
```
"""
function saddlepoint(coeffs::AbstractVector{<:Real};
    atol::Real=1e-12)::AbstractVector{<:Real}
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
    denom = z₀₁₁^2+(z₁₀₁-z₁₁₀)^2+z₀₂₀*(2z₁₀₁-z₂₀₀)-2z₀₁₁*(z₁₀₁+z₁₁₀-z₂₀₀)-z₀₀₂*(z₀₂₀-2z₁₁₀+z₂₀₀)
    
    if isapprox(denom,0,atol=atol)
        return [Inf,Inf,Inf]
    end
    sₑ = z₀₁₁^2+z₀₂₀*z₁₀₁+z₀₀₂*(-z₀₂₀+z₁₁₀)-z₀₁₁*(z₁₀₁+z₁₁₀)
    tₑ = -z₁₀₁*(z₀₁₁-z₁₀₁+z₁₁₀)+z₀₀₂*(z₁₁₀-z₂₀₀)+z₀₁₁*z₂₀₀
    uₑ = -(z₀₁₁+z₁₀₁-z₁₁₀)*z₁₁₀+z₀₂₀*(z₁₀₁-z₂₀₀)+z₀₁₁*z₂₀₀
    [sₑ,tₑ,uₑ]/denom
end

@doc """
    simplex_size(simplex)
Calculate the size of the region within a simplex.

# Arguments
- `simplex::AbstractMatrix{<:Real}`: the vertices of the simplex as columns of 
    an array.

# Returns
- `::Real`: the size of the region within the simplex. For example, the area
    within a triangle in 2D.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: simplex_size
simplex = [0 0 1; 0 1 0]
simplex_size(simplex)
# output
0.5
```
"""
function simplex_size(simplex::AbstractMatrix{<:Real})::Real
    abs(1/factorial(size(simplex,1))*det(vcat(simplex,ones(1,size(simplex,2)))))
end

@doc """
    insimplex(bpt;atol)

Check if a point lie within a simplex (including the boundary).

# Arguments
- `bpt::AbstractMatrix{<:Real}`: a point in Barycentric coordinates.
- `atol::Real=1e-12`: an absolute tolerance.

# Returns
- `::Bool`: a boolean indicating if the point is within the simplex.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: insimplex
bpt = Array([0 1]')
insimplex(bpt)
# output
true
```
"""
function insimplex(bpt::AbstractVector{<:Real};atol::Real=1e-12)
    (isapprox(maximum(bpt),1,atol=atol) || maximum(bpt) < 1) &&
    (isapprox(minimum(bpt),0,atol=atol) || minimum(bpt) > 0) &&
    isapprox(sum(bpt),1,atol=atol)
end

@doc """
    insimplex(bpts,atol)

Check if an array of points in Barycentric coordinates lie within a simplex.

# Arguments
- `bpts::AbstractMatrix{<:Real}`: an arry of points in barycentric coordinates
    as columns of an array.
- `atol::Real=1e-12`: absolute tolerance.
"""
function insimplex(bpts::AbstractMatrix{<:Real},atol::Real=1e-12)
    all(mapslices(x->insimplex(x,atol=atol),bpts,dims=1))
end

@doc """
    split_bezsurf₁(bezpts,atol)

Split a Bezier surface once into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_triangle
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 0.0 1.0 -1.0 0.0]
split_bezsurf₁(bezpts)
# output
3-element Array{Array{Float64,2},1}:
 [0.0 0.5 … 0.0 -1.0; 0.6 0.3 … 0.0 0.0; 0.08000000000000002 -0.40000000000000013 … 1.0 0.0]
 [0.0 0.0 … -0.5 -1.0; 0.6 0.8 … 0.5 0.0; 0.08000000000000002 -2.7755575615628914e-17 … 1.0 0.0]
 [0.0 0.0 … 0.5 1.0; 0.6 0.8 … 0.5 0.0; 0.08000000000000002 -2.7755575615628914e-17 … -1.0 0.0]
```
"""
function split_bezsurf₁(bezpts::AbstractMatrix{<:Real},
    allpts::AbstractArray=[]; atol::Real=1e-12)::AbstractArray

    dim = 2
    deg = 2
    triangle = bezpts[1:2,corner_indices]
    coeffs = bezpts[end,:]
    pts = bezpts[1:2,:]
    simplex_bpts = sample_simplex(dim,deg)
    intersects = simplex_intersects(bezpts,atol=atol)
    spt = saddlepoint(coeffs,atol=atol)

    if intersects == [[],[],[]]
        if insimplex(spt)
            allpts = [pts barytocart(spt,triangle)]
        else
            allpts = [pts]
        end
    else
        allintersects = reduce(hcat,[i for i=intersects if i!=[]])
        if insimplex(spt)
            allpts = [pts barytocart(spt,triangle) allintersects]
        else
            allpts = [pts allintersects]
        end
    end
    allpts = unique_points(allpts,atol=atol)
    
    tri_ind = MiniQhull.delaunay(allpts)
    subtri = [order_vertices!(allpts[:,tri_ind[:,i]]) for i=1:size(tri_ind,2)]
    sub_pts = [barytocart(simplex_bpts,tri) for tri=subtri]
    sub_bpts = [carttobary(pts,triangle) for pts=sub_pts]
    sub_vals = [reduce(hcat, [eval_poly(sub_bpts[j][:,i],coeffs,dim,deg) 
        for i=1:6]) for j=1:length(subtri)]
    sub_coeffs = [getpoly_coeffs(v[:],simplex_bpts,dim,deg) for v=sub_vals]
    sub_bezpts = [[sub_pts[i]; sub_coeffs[i]'] for i=1:length(sub_coeffs)]
    sub_bezpts
end

@doc """
    split_bezsurf(bezpts;atol)

Split a Bezier surface into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_triangle
bezpts = [-0.09385488270304788 0.12248162346376468 0.3388181296305772 0.09890589198180941 0.315242398148622 0.2916666666666667; 0.9061451172969521 0.7836634938331875 0.6611818703694228 0.6266836697595872 0.5042020462958225 0.34722222222222227; 0.0 7.949933953535975 3.9968028886505635e-15 8.042737134030771 -5.792491135426262 -11.720219017094017]
split_bezsurf₁(bezpts)
# output
2-element Array{Array{Float64,2},1}:
 [0.1291676795676943 0.23399290459913574 … 0.12248162346376468 -0.09385488270304788; 0.5828106204960847 0.6219962454327537 … 0.7836634938331875 0.9061451172969521; -5.329070518200751e-15 -4.5330445462060594e-15 … 7.9499339535359725 0.0]
 [0.1291676795676943 0.2104171731171805 … 0.315242398148622 0.3388181296305772; 0.5828106204960847 0.46501642135915344 … 0.5042020462958225 0.6611818703694228; -5.329070518200751e-15 -3.39004820851129 … -5.792491135426261 -1.1479627341393213e-15]
```
"""
function split_bezsurf(bezpts::AbstractMatrix{<:Real};atol=1e-12)::AbstractArray
    
    intersects = simplex_intersects(bezpts,atol=atol)
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects <= 2
        return [bezpts]
    else
        sub_bezpts = split_bezsurf₁(bezpts)
        sub_intersects = [simplex_intersects(b) for b=sub_bezpts]
        num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
            size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
        while any(num_intersects .> 2)
            for i = length(num_intersects):-1:1
                if num_intersects[i] <= 2 continue end
                append!(sub_bezpts,split_bezsurf₁(sub_bezpts[i]))                
                deleteat!(sub_bezpts,i)
                sub_intersects = [simplex_intersects(b) for b=sub_bezpts]
                num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
                    size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
            end
        end
    end
    sub_bezpts
end

@doc """
    analytic_area(w::Real)

Calculate the area within a triangle and a canonical, rational, Bezier curve.

# Arguments
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic, Bezier curve.
# Returns
- `::Real`: the area within the triangle and Bezier curve.

# Examples
```jldoctest
w = 1.0
analytic_area(w)
# output
0.6666666666666666
```
"""
function analytic_area(w::Real)::Real
    
    # Use the Taylor expansion of the analytic expression if the weight is close to 1.
    if isapprox(w,1,atol=1e-2)
        2/3+4/15*(-1+w)-6/35*(-1+w)^2+32/315*(-1+w)^3-40/693*(-1+w)^4+(32*(-1+w)^5)/1001-
        (112*(-1+w)^6)/6435+ (1024*(-1+w)^7)/109395-(1152*(-1+w)^8)/230945+
        (2560*(-1+w)^9)/969969-(2816*(-1+w)^10)/2028117
    else
        a = sqrt(Complex(-1-w))
        b = sqrt(Complex(-1+w))
        abs(real((w*(w+(2*atan(b/a)/(a*b)))/(-1+w^2))))
    end
end

@doc """
    analytic_coeffs(coeffs,w;atol=1e-12)

Calculate the volume within a canonical triangle and Bezier curve of a quadratic surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratica surface.
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic, Bezier curve.
- `atol::Real=1e-12`: an absolute tolerance for finite precision tolerances.

# Returns
- `::Real`: the area within the triangle and Bezier curve.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: analytic_volume
coeffs = [0.2,0.2,0.3,-0.3,0.4,-0.4]
w = 0.3
analytic_volume(coeffs,w)
# output
0.4426972170733675
```
"""
function analytic_volume(coeffs::AbstractVector{<:Real},w::Real;
        atol::Real=1e-12)::Real
    
    (c₅,c₃,c₀,c₄,c₁,c₂) = coeffs
    d = c₀+c₁+c₂+c₃+c₄+c₅
    #if isapprox(d,0,atol=atol)
    #    return 0
    #end
    # Use the Taylor expansion of the analytic solution if the weight is close to 1.
    if isapprox(w,1,atol=1e-2)
        d/6*((6/7+(2*(-11*c₀-5*(c₁+c₃)+c₄))/(35*d))+4/105*(5+(3*c₀+5*(c₁+c₃)-c₄)/d)*(w-1)+(-(2/11)+(2*(81*c₀+
        5*(-5*(c₁+c₃)+c₄)))/(1155*d))*(w-1)^2+(32*(70+(-89*c₀+5*(-5*(c₁+c₃)+c₄))/d)*(w-1)^3)/15015+
        (8*(17*c₀-7*(c₁+6*c₂+c₃+7*c₄+6*c₅))*(w-1)^4)/(3003*d)+(64*(315+(-432*c₀+77*(-5*(c₁+c₃)+c₄))/
        d)*(w-1)^5)/255255+(224*(43*c₀+3*(30*c₁-55*c₂+30*c₃-72*c₄-55*c₅))*(w-1)^6)/(692835*d)+
        (1024*(165-(4*(46*c₀+75*(c₁+c₃)-15*c₄))/d)*(w-1)^7)/4849845-(384*(93*c₀-55*(41*c₁-39*c₂+41*c₃-55*c₄-39*c₅))*(w-1)^8)/
        (37182145*d)+(512*(1001+(-797*c₀+451*(-5*(c₁+c₃)+c₄))/d)*(w-1)^9)/37182145-
        (2816*(164*c₀+13*(-50*c₁+35*c₂-50*c₃+52*c₄+35*c₅))*(w-1)^10)/(152108775*d))
    else
        a = sqrt(Complex(-1-w))
        b = sqrt(Complex(-1+w))
#        sign(w)real((w*(a*b*w*(-32*c₁+33*c₂-32*c₃+46*c₄+33*c₅-2*(-26*c₀+18*c₁+13*c₂+18*c₃+12*c₄+13*c₅)*w^2+
#            8*d*w^4)+6*(5*c₂+6*c₄+5*c₅+4*(c₀-5*(c₁+c₃)+c₄)*w^2+16*c₀*w^4)*atan(b/a)))/(8*d*a*b*(-1+w^2)^3))
        sign(w)real((w*(a*b*w*(-32*c₁+33*c₂-32*c₃+46*c₄+33*c₅-2*(-26*c₀+18*c₁+13*c₂+18*c₃+12*c₄+13*c₅)*w^2+
            8*d*w^4)+6*(5*c₂+6*c₄+5*c₅+4*(c₀-5*(c₁+c₃)+c₄)*w^2+16*c₀*w^4)*atan(b/a)))/(6*8*a*b*(-1+w^2)^3))

    end
end

@doc """
    sub₋coeffs(bezpts,subtriangle)

Calculate the coefficients of a quadratic sub-surface of a quadratic triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic triangle.
- `subtriangle::AbstractMatrix{<:Real}`: a subtriangle give by the points at
    its corners as columns of an array.

# Returns
- `::AbstractVector{<:Real}`: the coefficients of the quadratic triangle over a
    sub-surface of a quadratic triangle.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: sub₋coeffs
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.25 -0.25 3.75 -0.25 1.75 1.75]
subtriangle = [-0.5 0.0 -0.6464466094067263; 0.0 1.0 0.35355339059327373]
sub₋coeffs(bezpts,subtriangle)
# output
6-element Array{Float64,1}:
  0.0
  0.25
  1.75
 -0.07322330470336313
  0.45710678118654746
 -5.551115123125783e-17
```
"""
function sub₋coeffs(bezpts::AbstractMatrix{<:Real},
    subtriangle::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    ptsᵢ = carttobary(barytocart(sample_simplex(2,2),subtriangle),bezpts[1:2,corner_indices])
    valsᵢ = eval_poly(ptsᵢ,bezpts[end,:],2,2)
    getpoly_coeffs(valsᵢ,sample_simplex(2,2),2,2)
end

@doc """
    two₋intersects_area₋volume(bezpts,quantity,intersects=[];atol=1e-12)

Calculate the area or volume within a quadratic curve and triangle and Quadratic surface.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of a quadratic surface.
- `quantity::String`: the quantity to compute ("area" or "volume").
- `intersects::AbstractArray=[]`: the two point where the curve intersects the 
    triangle as columns of an array.
- `atol::Real`: an absolute tolerance.

# Returns
- `areaₒᵣvolume::Real`: the area within the curve and triangle or the volume below the surface
    within the curve and triangle. The area is on the side of the curve where the surface is 
    less than zero.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: two₋intersects_area₋volume
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.89 -0.08 -1.28 1.12 -0.081 -0.88]
two₋intersects_area₋volume(bezpts,"volume")
# output
-0.3533719907367465
```
"""
function two₋intersects_area₋volume(bezpts::AbstractMatrix{<:Real},
    quantity::String; atol::Real=1e-12)::Real

    # Calculate the bezier curve and weights make sure the curve passes through
    # the triangle
    triangle = bezpts[1:2,corner_indices]
    coeffs = bezpts[end,:]
    intersects = simplex_intersects(bezpts,atol=atol)
    bezptsᵣ = [0;0]
    if intersects != [[],[],[]]
        all_intersects = reduce(hcat,[i for i=intersects if i!= []])
        if size(all_intersects,2) != 2
            error("Can only calculate the area or volume when the curve intersects 
                the triangle at two points or doesn't intersect the triangle.")
        end
        p₀ = all_intersects[:,1]
        p₂ = all_intersects[:,2]
        (bezptsᵣ,bezwtsᵣ) = getbez_pts₋wts(bezpts,p₀,p₂,atol=atol)
        ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
        # Make sure the weight of the middle Bezier point has the correct sign.
        if !insimplex(carttobary(ptᵣ,triangle),atol=atol)
            bezwtsᵣ[2] *= -1
            ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
            if !insimplex(carttobary(ptᵣ,triangle),atol=atol)
                intersects = [[],[],[]]
            end
        end
    end

    # If the tangent lines are close to parallel, the middle Bezier point of the
    # curve will be very far away, which introduces numerical errors. We handle
    # this by splitting the surface up and recalculating.
    # Also, split the surface if the level curve isn't linear and the saddle point 
    # is within the triangle.
    cstype = conicsection(bezpts[end,:],atol=atol)
    linear = any(cstype .== ["line","rectangular hyperbola","parallel lines"])
    if maximum(abs.(bezptsᵣ)) > 1e6 || (insimplex(saddlepoint(bezpts[end,:],atol=atol),atol=atol) && !linear) 
        bezptsᵤ = [split_bezsurf(b,atol=atol) for b=split_bezsurf₁(bezpts)] |> flatten |> collect
        return sum([two₋intersects_area₋volume(b,quantity,atol=atol) for b=bezptsᵤ])
    end

    # No intersections
    if intersects == [[],[],[]]
        if all(bezpts[end,corner_indices] .< 0 .| 
            isapprox.(bezpts[end,corner_indices],0,atol=atol))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
        else
            areaₒᵣvolume = 0
        end
        return areaₒᵣvolume
    end

    edgesᵢ = [i for i=1:3 if intersects[i] != []]
    if length(edgesᵢ) == 1
        # When intersections are on two different edges, we need to include the
        # area or volume from a subtriangle in addition to the canonical rational
        # Bezier triangle and the whole triangle. It has no effect when the intersections
        # are on the same edge.
        corner = [1,2,3][edgesᵢ[1]]
        # Determine which region to keep from the opposite corner.
        opp_corner = [6,1,3][edgesᵢ[1]]
    elseif length(edgesᵢ) ==2
        corner = [3,1,2][setdiff([1,2,3],edgesᵢ)[1]]
        cornersᵢ = sort(unique([isapprox(all_intersects[:,j],triangle[:,i],
            atol=atol) ? i : 0 for i=1:3,j=1:2]))
        if cornersᵢ != [0] && length(cornersᵢ) == 3
            # Case where intersections are at two corners.
            opp_corner = [1,3,6][(setdiff([1,2,3],cornersᵢ[2:end])[1])]
        elseif (cornersᵢ != [0] && length(cornersᵢ) == 2) || cornersᵢ == [0]
            # Case where there the intersection are on adjacent edges of the
            # the triangle and neither are at corners or one at corner.
            opp_corner = [1,3,6][(setdiff([1,2,3],edgesᵢ)[1])]
            corner = [3,1,2][setdiff([1,2,3],edgesᵢ)[1]]
        else
            error("The intersections may only intersect at most two corners.")
        end
    else
        error("The curve may only intersect at most two edges.")
    end

    simplex_bpts = sample_simplex(2,2)
    triangleₑ = order_vertices!([all_intersects triangle[:,corner]])
    if quantity == "area"
        # curve area or volume
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_area(bezwtsᵣ[2])
    elseif quantity == "volume"
        coeffsᵣ = sub₋coeffs(bezpts,bezptsᵣ)
        #areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*mean(coeffsᵣ)*analytic_volume(coeffsᵣ,bezwtsᵣ[2],atol=atol)
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_volume(coeffsᵣ,bezwtsᵣ[2],atol=atol)
    else
        throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
    end

    # Get the sign of the area correct (accounts for the curvature of the curve).
    inside = false
    # Get exception when corners of triangleₑ all lie on a straight line.
    try
        inside = insimplex(carttobary(ptᵣ,triangleₑ),atol=atol)
    catch SingularException
        nothing
    end

    if length(edgesᵢ) == 2 && inside
        areaₒᵣvolumeᵣ *= -1
    end

    below₀ = bezpts[end,opp_corner] < 0 || isapprox(bezpts[end,opp_corner],0,atol=atol)
    if quantity == "area"
        areaₒᵣvolume =  areaₒᵣvolumeᵣ + simplex_size(triangleₑ)
        if below₀
            areaₒᵣvolume = simplex_size(triangle) - areaₒᵣvolume
        end
    else # quantity == "volume"
        coeffsₑ = sub₋coeffs(bezpts,triangleₑ)
        areaₒᵣvolume = mean(coeffsₑ)*simplex_size(triangleₑ) + areaₒᵣvolumeᵣ
        if below₀
            areaₒᵣvolume = simplex_size(triangle)*mean(coeffs) - areaₒᵣvolume
        end
    end

    areaₒᵣvolume
end

@doc """
    quad_area₋volume(bezpts,quantity;atol=1e-12)

Calculate the area of the shadow of a quadric or the volume beneath the quadratic.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `quantity::String`: the quantity to calculate ("area" or "volume").
- `atol::Real=1e-12`: an absolute tolerance for floating point comparisons.

# Returns
- `::Real`: the area of the shadow of a quadratic polynomial within a triangle
    and below the plane `z=0` or the volume of the quadratic polynomial under the 
    same constraints.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: quad_area₋volume
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2/3 -4/3 2/3 -2/3 -2/3 0]
quad_area₋volume(bezpts,"area")
# output
0.869605101106897
```
"""
function quad_area₋volume(bezpts::AbstractMatrix{<:Real},
        quantity::String;atol::Real=1e-12)::Real
    sum([two₋intersects_area₋volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])    
end

@doc """
    calc_mesh₋bezcoeffs(recip_latvecs,rules,cutoff,sheets,mesh,energy_conversion_factor; rtol,atol)

Calculate the quadratic coefficients of a triangulation of the IBZ.

# Arguments
- `recip_latvecs::AbstractMatrix{<:Real}`: the reciprocal lattice basis as columns of
    a square matrix.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets<:Int`: the number of sheets considered in the calculation.
- `mesh::Delaunay.Triangulation`: a simplex tesselation of the IBZ.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(recip_latvecs))))`: a relative tolerance for
    finite precision comparisons. This is used for identifying points within a
    circle or sphere in the Fourier expansion of the EPM.
- `atol::Real=1e-9`: an absolute tolerance for finite precision comparisons.

# Output
- `mesh_bezcoeffs::Vector{Vector{Any}}`: the coefficients of all quadratic polynomials. The
    array is ordered first by `simplex` and then by `sheet`: `mesh_bezcoeffs[simplex][sheet]`.

# Examples
```jldoctest
import Pebsi.EPMs: m4recip_latvecs, m4rules, m4cutoff, m4ibz
import Delaunay: delaunay
sheets = 2
mesh = delaunay(m4ibz.points)
energy_conv = 1
mesh_bezcoeffs = calc_mesh₋bezcoeffs(m4recip_latvecs,m4rules,m4cutoff,sheets,mesh,energy_conv)
# output
2-element Vector{Vector{Any}}:
 [[0.4825655582645329, 0.46891799288429503, 0.45695660389445203, -0.20577513367855282, -0.2284287599373549, -0.319970723890622], [1.0021520603113079, 0.962567754290957, 0.9364044831997849, 0.9050494036379049, 1.5874293259883903, 1.041804108772328]]
 [[-0.28153999982153577, -0.18566890981787773, 0.4825655582645329, -0.30280441786109924, -0.2057751336785528, -0.319970723890622], [0.5806033720376905, 0.8676008216346605, 1.0021520603113079, 0.6209049649780336, 0.905049403637905, 1.041804108772328]]
```
"""
function calc_mesh₋bezcoeffs(recip_latvecs::AbstractMatrix{<:Real},
    rules::Dict{Float64,Float64},cutoff::Real,sheets::Int,mesh::Triangulation,
    energy_conversion_factor::Real=RytoeV; 
    rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),atol::Real=1e-9)::Vector{Vector{Any}}

    dim,deg=(2,2)
    simplex_bpts=sample_simplex(dim,deg)
    mesh_bezcoeffs = [Vector{Any}[] for i=1:size(mesh.simplices,1)]
    for s = 1:size(mesh.simplices,1)
        simplex = Array(mesh.points[mesh.simplices[s,:],:]')
        simplex_pts = barytocart(simplex_bpts,simplex)
        values = eval_epm(simplex_pts,recip_latvecs,rules,cutoff,sheets,
            energy_conversion_factor,rtol=rtol,atol=atol)
        mesh_bezcoeffs[s] = [getpoly_coeffs(values[i,:],simplex_bpts,dim,deg) for i=1:sheets]
    end
    mesh_bezcoeffs
end

@doc """
    shadow₋size(recip_latvecs,rules,cutoff,sheets,mesh,fermi_level,energy_conversion_factor;rtol,atol)

Calculate the size of the shadow of the band structure beneath the Fermi level.

# Arguments
- `recip_latvecs::AbstractMatrix{<:Real}`: the reciprocal lattice basis as columns of
    a square matrix.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets<:Int`: the number of sheets considered in the calculation.
- `mesh::Delaunay.Triangulation`: a simplex tesselation of the IBZ.
- `fermi_level::Real`: an estimate of the value of the Fermi level.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(recip_latvecs))))`: a relative tolerance for
    finite precision comparisons. This is used for identifying points within a
    circle or sphere in the Fourier expansion of the EPM.
- `atol::Real=1e-9`: an absolute tolerance for finite precision comparisons.

# Returns
    `shadow_size::Real`: the size of the shadow of the sheets onto the IBZ for the 
    estimated Fermi level.

# Examples
```jldoctest
import Pebsi.EPMs: m5recip_latvecs, m5real_latvecs, m5rules, m5cutoff, m5ibz
import Delaunay: delaunay
sheets = 5
mesh = delaunay(m5ibz.points)
fermi_level = 0.5
energy_conv=1
shadow₋size(m5recip_latvecs,m5rules,m5cutoff,sheets,mesh,fermi_level,energy_conv)
# output
0.9257020287626774
```
"""
function shadow₋size(recip_latvecs::AbstractMatrix{<:Real},rules::Dict{Float64,Float64},
        cutoff::Real,sheets::Int,mesh::Triangulation,fermi_level::Real,
        energy_conversion_factor::Real=RytoeV;rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),atol::Real=1e-9)::Real
    
    dim,deg=(2,2)
    simplex_bpts=sample_simplex(dim,deg)
    shadow_size = 0
    for s = 1:size(mesh.simplices,1)
        simplex = Array(mesh.points[mesh.simplices[s,:],:]')
        simplex_pts = barytocart(simplex_bpts,simplex)
        values = eval_epm(simplex_pts,recip_latvecs,rules,cutoff,sheets,energy_conversion_factor) .- fermi_level
        sheet_bezpts = [[simplex_pts; getpoly_coeffs(values[i,:],simplex_bpts,dim,deg)'] for i=1:sheets[end]]
        shadow_size += sum([quad_area₋volume(sheet_bezpts[i],"area") for i=1:sheets[end]])
    end
    shadow_size
end


function shadow₋size(mesh::Triangulation, mesh_bezcoeffs::Vector{Vector{Any}},fermi_level::Real)::Real

    dim,deg=(2,2)
    simplex_bpts=sample_simplex(dim,deg)
    sheets = size(mesh_bezcoeffs[1],1)
    shadow_size = 0
    for s = 1:size(mesh.simplices,1)
        simplex = Array(mesh.points[mesh.simplices[s,:],:]')
        simplex_pts = barytocart(simplex_bpts,simplex)
        sheet_bezpts = [Matrix{Real}([simplex_pts; mesh_bezcoeffs[s][i]']) for i=1:sheets]
        shadow_size += sum([quad_area₋volume(sheet_bezpts[i],"area") for i=1:sheets])
    end
    shadow_size
end

@doc """
    get₋neighbors(index,mesh,num₋neighbors=2)

Calculate the nth-nearest neighbors of a point in a mesh.

# Arguments
- `index::Int`: the index of the point in the mesh. The coordinates
    of the point are `mesh.points[index,:]`.
- `mesh::Triangulation`: a Delaunay triangulation of the mesh from 
    `Delaunay.delaunay`.
- `num₋neighbors::Int=2`: the number of neighbors to find. For example,
    if 2, find first and second nearest neighbors.

# Returns
- `indices::AbstractArray{Int,1}`: the indices of neighboring points.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: get₋neighbors
import Delaunay: delaunay
pts = [0.0 0.0; 0.25 0.0; 0.5 0.0; 0.25 0.25; 0.5 0.25; 0.5 0.5]
index = 2
mesh = delaunay(pts)
get₋neighbors(index,mesh)
# output
5-element Array{Int64,1}:
 4
 1
 5
 3
 6
```
"""
function get₋neighbors(index::Int,mesh::Triangulation,num₋neighbors::Int=2)::AbstractArray{Int,1}
    indices = [index]
    for _=1:num₋neighbors
        (rows,cols,) = findnz(mesh.vertex_neighbor_vertices)
        first₋neighbors = rows[findall(x->any(x .== indices),cols)]
        indices = unique([indices;first₋neighbors])
    end
    indices[2:end]
end

@doc """
    lineseg₋pt_dist(line_seg,pt)

Calculate the shortest distance from a point to a line segment.

# Arguments
- `line_seg::Matrix{<:Real}`: a line segment given by two points. The points
    are the columns of the matrix.
- `p3::Vector{<:Real}`: a point in Cartesian coordinates.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `d::Real`: the shortest distance from the point to the line segment.

# Examples
```jldoctest
lineseg = [0 0; 0 1]
pt = [0.5, 0.5]
lineseg₋pt_dist(lineseg,pt)
# output
0.5000000000000001
```
"""
function lineseg₋pt_dist(line_seg::Matrix{<:Real},p3::Vector{<:Real};atol::Real=1e-9)::Real
    
    p1 = line_seg[:,1]
    p2 = line_seg[:,2]
    proj = dot((p2-p1)/norm(p2 - p1),p3-p1)
    if proj <= norm(p2 - p1) && 0 <= proj
        if isapprox(norm(p3 - p1)^2 - proj^2,0,atol=atol)
            d = 0
        else
            d = √(norm(p3 - p1)^2 - proj^2)
        end
    else
        d = minimum([norm(p3 - p1),norm(p3 - p2)])
    end 
end

end # module