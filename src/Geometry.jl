module Geometry

using ..Defaults: def_atol
using LinearAlgebra: dot,cross,norm,det
using Base.Iterators: product

export order_vertices!, sample_simplex, barytocart, carttobary, simplex_size, 
    insimplex, lineseg₋pt_dist, affine_trans, mapto_xyplane, ptface_mindist

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
    sample_simplex(dim,deg)

Get the sample points of a simplex for a polynomial approximation.

# Arguments
- `dim::Integer`: the number of dimensions (2 = triangle, 3 = tetrahedron).
- `deg::Integer`: the degree of the polynomial approximations.

# Returns
- `::AbstractMatrix{<:Real}`: the points on the simplex as columns of an array.

# Examples
```jldoctest
import Pebsi.Polynomials: sample_simplex
dim = 2
deg = 1
sample_simplex(dim,deg)
# output
3×3 Array{Float64,2}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
"""
function sample_simplex(dim::Integer,deg::Integer,
    rtol::Real=sqrt(eps(1.0)))::AbstractMatrix{<:Real}
    reduce(hcat,filter(x->length(x)>0, 
        [isapprox(sum(p),1,rtol=rtol,atol=def_atol) ? collect(p) : [] 
        for p=collect(product([0:1/deg:1 for i=0:dim]...))]))
end

@doc """
    barytocart(barypt,simplex)

Convert a point from barycentric to Cartesian coordinates.

# Arguments
- `barypt::AbstractVector{<:Real}`: a point in Barycentric coordinates.
- `simplex::AbstractMatrix{<:Real}`: the vertices of a simplex as columns of
    an array.

# Returns
- `::AbstractVector{<:Real}`: the point in Cartesian coordinates.

# Examples
```jldoctest
import Pebsi.Polynomials: barytocart
barypt = [0,0,1]
simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
barytocart(barypt,simplex)
# output
2-element Array{Float64,1}:
 0.5
 0.0
```
"""
function barytocart(barypt::AbstractVector{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    simplex*barypt
end

@doc """
    barytocart(barypts,simplex)

Convert points as columns on an array from barycentric to Cartesian coordinates.
"""
function barytocart(barypts::AbstractMatrix{<:Real},
    simplex::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    mapslices(x->barytocart(x,simplex),barypts,dims=1)
end

@doc """
    carttobary(pt,simplex)

Transform a point from Cartesian to barycentric coordinates.

# Arguments
- `pt::AbstractVector{<:Real}`: the point in Cartesian coordinates
- `simplex::AbstractMatrix{<:Real}`: the corners of the simplex as columns of an array.
"""
function carttobary(pt::AbstractVector{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractVector{<:Real}

    inv(vcat(simplex,ones(Int,(1,size(simplex,2)))))*vcat(pt,1)
end

@doc """
    carttobary(pt,simplex)

Transform an array of points from Cartesian to barycentric coordinates.

# Arguments
- `pts::AbstractMatrix{<:Real}`: the points in Cartesian coordinates as columns of an array.
"""
function carttobary(pts::AbstractMatrix{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    mapslices(x->carttobary(x,simplex),pts,dims=1)
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
- `atol::Real=1e-9`: an absolute tolerance.

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
function insimplex(bpt::AbstractVector{<:Real};atol::Real=def_atol)
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
- `atol::Real=1e-9`: absolute tolerance.
"""
function insimplex(bpts::AbstractMatrix{<:Real},atol::Real=def_atol)
    all(mapslices(x->insimplex(x,atol=atol),bpts,dims=1))
end

@doc """
    lineseg₋pt_dist(line_seg,pt)

Calculate the shortest distance from a point to a line segment.

# Arguments
- `line_seg::AbstractMatrix{<:Real}`: a line segment given by two points. The points
    are the columns of the matrix.
- `p3::AbstractVector{<:Real}`: a point in Cartesian coordinates.
- `line::Bool=false`: if true, calculate the distance from a line instead of a 
    line segment.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `d::Real`: the shortest distance from the point to the line segment.

# Examples
```jldoctest
import Pebsi.Geometry: lineseg₋pt_dist
lineseg = [0 0; 0 1]
pt = [0.5, 0.5]
lineseg₋pt_dist(lineseg,pt)
# output
0.5000000000000001
```
"""
function lineseg₋pt_dist(p3::AbstractVector{<:Real},line_seg::AbstractMatrix{<:Real},
    line::Bool=false;atol::Real=def_atol)::Real
    
    p1 = line_seg[:,1]; p2 = line_seg[:,2]
    proj = dot((p2-p1)/norm(p2 - p1),p3-p1)

    if proj <= norm(p2 - p1) && 0 <= proj || line
        if isapprox(norm(p3 - p1)^2 - proj^2,0,atol=atol)
            d = 0
        else
            d = √(norm(p3 - p1)^2 - proj^2)
        end
    else
        d = minimum([norm(p3 - p1),norm(p3 - p2)])
    end 
end


@doc """
    ptface_mindist(pt,face)

Calculate the minimum distance between a point and a finite plane.

# Arguments
- `pt`: the 3D Cartesian coordinates of a point embedded in 3D.
- `face`: the 3D Cartesian coordinates of the corners of a face that lie on the
    same plane as columns of a matrix.

# Returns
- The minimum distance from the point to the plane.
"""
function ptface_mindist(pt,face)
    # Use points that are not colinear to find a normal vector perpendicular to the face.

    i = 3
    v₁ = face[:,2] - face[:,1]
    v₂ = face[:,i] - face[:,1]
    n = cross(v₁,v₂)
    while all(isapprox.(n,0,atol=def_atol))
        i += 1
        if i > size(face,2)
            error("All points of the face are collinear.")
        end
        v₂ = face[:,i] - face[:,1]
        n = cross(v₁,v₂)
    end

    n = n/norm(n)
    minimum([[norm(pt - face[:,i]) for i=1:size(face,2)]; abs(dot(pt - face[:,1],n))])
end

corner_indices = [1,3,6]

@doc """
    affine_trans(pts)

Calculate the affine transformation that maps points to the xy-plane.

# Arguments
- `pts::AbstractMatrix{<:Real}`: Cartesian points as the columns of a matrix.
    The points must all lie on a plane in 3D.

# Returns
- `M::AbstractMatrix{<:Real}`: the affine transformation matrix that operates
    on points in homogeneous coordinates from the left.

# Examples
```jldoctest
using SymmetryReduceBZ
pts = [0.5 0.5 0.5; 0.5 -0.5 0.5; -0.5 0.5 0.5; -0.5 -0.5 0.5]'
SymmetryReduceBZ.Utilities.affine_trans(pts)
# output
4×4 Matrix{Float64}:
  0.0  -1.0   0.0  0.5
 -1.0   0.0   0.0  0.5
  0.0   0.0  -1.0  0.5
  0.0   0.0   0.0  1.0
```
"""
function affine_trans(pts::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    a,b,c = [pts[:,i] for i=corner_indices]

    # Create a coordinate system with two vectors lying on the plane the points
    # lie on.
    u = b-a
    v = c-a
    u = u/norm(u)
    v = v - dot(u,v)*u/dot(u,u)
    v = v/norm(v)
    w = cross(u,v)

    # Augmented matrix of affine transform
    inv(vcat(hcat([u v w],a),[0 0 0 1]))
end

@doc """
    function mapto_xyplane(pts)

Map Cartesian points embedded in 3D on a plane to the xy-plane embedded in 2D.

# Arguments
- `pts::AbstractMatrix{<:Real}`: Cartesian points embedded in 3D as columns of a
    matrix.

# Returns
- `AbstractMatrix{<:Real}`: Cartesian points in 2D as columns of a matrix.

# Examples
```jldoctest
using SymmetryReduceBZ
pts = [0.5 -0.5 0.5; 0.5 -0.5 -0.5; 0.5 0.5 -0.5; 0.5 0.5 0.5]'
SymmetryReduceBZ.Utilities.mapto_xyplane(pts)
# output
2×4 Matrix{Float64}:
 0.0  1.0  1.0  0.0
 0.0  0.0  1.0  1.0
```
"""
function mapto_xyplane(pts::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}

    M = affine_trans(pts)
    reduce(hcat,[(M*[pts[:,i]..., 1])[1:2] for i=1:size(pts,2)])
end

end # Geometry