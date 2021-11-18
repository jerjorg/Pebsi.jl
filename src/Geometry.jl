module Geometry

using ..Defaults: def_atol
using LinearAlgebra: dot, cross, norm, det
using Base.Iterators: product

export order_vertices!, sample_simplex, barytocart, carttobary, simplex_size, 
    insimplex, lineseg₋pt_dist, affine_trans, mapto_xyplane, ptface_mindist,
    point_in_polygon

@doc """
    order_vertices(vertices)

Put the vertices of a triangle in counterclockwise order.

# Arguments
- `vertices::AbstractMatrix{<:Real}`: the vertices of the triangle in columns of
    a matrix.

# Returns
- `vertices::AbstractMatrix{<:Real}`: the vertices of the triangle in columns of
    a matrix where increasing the column number moves around the triangle
    counterclockwise.

# Examples
```jldoctest
using Pebsi.Geometry: order_vertices!
triangle = [0 1 1; 0 1 0]
order_vertices!(triangle)
# output
2×3 Matrix{Int64}:
 0  1  1
 0  0  1
```
"""
function order_vertices!(vertices::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
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
- `::AbstractMatrix{<:Real}`: the points on the simplex in Cartesian coordinates
    as columns of a matrix.

# Examples
```jldoctest
import Pebsi.Polynomials: sample_simplex
dim = 2
deg = 1
sample_simplex(dim,deg)
# output
3×3 Matrix{Float64}:
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
- `barypt::AbstractVector{<:Real}`: a point in barycentric coordinates.
- `simplex::AbstractMatrix{<:Real}`: the vertices of a simplex as columns of
    an array in Cartesian coordinates.

# Returns
- `::AbstractVector{<:Real}`: the point in Cartesian coordinates.

# Examples
```jldoctest
import Pebsi.Polynomials: barytocart
barypt = [0,0,1]
simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
barytocart(barypt,simplex)
# output
2-element Vector{Float64}:
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

Convert points from barycentric to Cartesian coordinates.

# Arguments
- `barypts::AbstractMatrix{<:Real}`: a matrix of points in barycentric
    coordinates as columns.
- `simplex::AbstractMatrix{<:Real}`: the Cartesian coordinates of the corners of
    a triangle.

# Examples
```jldoctest
using Pebsi.Geometry: barytocart
pts = [0 0; 0 1; 1 0; 0 0]
triangle = [0 1 1; 0 0 1]
barytocart(pts,triangle)
# output
2×2 Matrix{Float64}:
 0.5  1.0
 0.0  0.0
```
"""
function barytocart(barypts::AbstractMatrix{<:Real},
    simplex::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    mapslices(x->barytocart(x,simplex),barypts,dims=1)
end

@doc """
    carttobary(pt,simplex)

Transform a point from Cartesian to barycentric coordinates.

# Arguments
- `pt::AbstractVector{<:Real}`: the point in Cartesian coordinates.
- `simplex::AbstractMatrix{<:Real}`: the corners of the simplex as columns of an array.

# Returns
- `::AbstractVector{<:Real}`: the same point in barycentric coordinates.

# Examples
```jldoctest
using Pebsi.Geometry: carttobary
triangle = [0. 1. 1.; 0. 0. 1.]
pt = [0.5,0.5]
carttobary(pt,triangle)
# output
3-element Vector{Float64}:
 0.5
 0.0
 0.5
```
"""
function carttobary(pt::AbstractVector{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractVector{<:Real}

    inv(vcat(simplex,ones(Int,(1,size(simplex,2)))))*vcat(pt,1)
end

@doc """
    carttobary(pts,simplex)

Transform points from Cartesian to barycentric coordinates.

# Arguments
- `pts::AbstractMatrix{<:Real}`: the points in Cartesian coordinates as columns 
    of a matrix.
- `simplex::AbstractMatrix{<:Real}`: the Cartesian coordinates of the corners of
    a triangle as columns of a matrix.

# Output
- `::AbstractMatrix{<:Real}`: the same provided points in barycentric coordinates
    in columns of a matrix.
# Examples
```jldoctest
using Pebsi.Geometry: carttobary
pts = [0.25 0.5; 0.25 0.5]
triangle = [0. 1. 1.; 0. 0. 1.]
carttobary(pts,triangle)
# output
3×2 Matrix{Float64}:
 0.75  0.5
 0.0   0.0
 0.25  0.5
```
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
- `bpt::AbstractMatrix{<:Real}`: a point in barycentric coordinates.
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

Check if an array of points in barycentric coordinates lie within a simplex.

# Arguments
- `bpts::AbstractMatrix{<:Real}`: an arry of points in barycentric coordinates
    as columns of an array.
- `atol::Real=1e-9`: an absolute tolerance.

# Returns
- `Bool`: is `true` if all the points lie within the simplex

# Examples
```jldoctest
using Pebsi.Geometry: insimplex
bpts = [0 0; 0 1; 1 0]
insimplex(bpts)
# output
true
```
"""
function insimplex(bpts::AbstractMatrix{<:Real},atol::Real=def_atol)
    all(mapslices(x->insimplex(x,atol=atol),bpts,dims=1))::Bool
end

@doc """
    lineseg₋pt_dist(p3,line_seg,line;atol)

Calculate the shortest distance from a point to a line segment.

# Arguments
- `p3::AbstractVector{<:Real}`: a point in Cartesian coordinates.
- `line_seg::AbstractMatrix{<:Real}`: a line segment given by two points. The points
    are the columns of the matrix.
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
lineseg₋pt_dist(pt,lineseg)
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
    point_in_polygon(pt,polygon)

Determine if a point lies within a polygon where both are embedded in 3D.

# Arguments
- `pt`: the Cartesial coordinates of a point that lies on the same plane as
    the polygon.
- `polygon`: the Cartesian coordinates of the corners of a polygon as columns of
    a matrix.

# Returns
- `Bool`: true if the point lines inside the triangle (including boundaries)

# Examples
```jldoctest
using Pebsi.Geometry: point_in_polygon
using Pebsi.Geometry: point_in_polygon
point = [0,0,1]
polygon = [-1 1 1 -1; -1 -1 1 1; 1 1 1 1]
point_in_polygon(point,polygon)
# output
true
```
"""
function point_in_polygon(pt,polygon; atol=def_atol)::Bool
    x,y,z=pt
    npts = size(polygon,2)
    s1 = zeros(npts); s2 = zeros(npts); s3 = zeros(npts)
    for i=1:npts
        x1,y1,z1 = polygon[:,i]
        x2,y2,z2 = polygon[:,mod1(i+1,npts)]
        s1[i] = (y - y1)*(x2 - x1) - (x - x1)*(y2-y1)
        s2[i] = (z - z1)*(x2 - x1) - (x - x1)*(z2-z1)
        s3[i] = (y - y1)*(z2 - z1) - (z - z1)*(y2-y1)
    end
    ((all((s1 .>= 0) .| isapprox.(s1,0,atol=atol)) | 
     all((s1 .<= 0) .| isapprox.(s1,0,atol=atol))) 
     &
    (all((s2 .>= 0) .| isapprox.(s2,0,atol=atol)) |
     all((s2 .<= 0) .| isapprox.(s2,0,atol=atol)))
     & 
    (all((s3 .>= 0) .| isapprox.(s3,0,atol=atol)) |
     all((s3 .<= 0) .| isapprox.(s3,0,atol=atol)))
     )
end 

@doc """
    ptface_mindist(pt,face)

Calculate the minimum distance between a point and a finite plane.

# Arguments
- `pt::AbstractVector{<:Real}`: the 3D Cartesian coordinates of a point embedded
    in 3D.
- `face::AbstractMatrix{<:Real}`: the 3D Cartesian coordinates of the corners of
    a face that lie on the same plane as columns of a matrix.

# Returns
- `::Real`: The minimum distance from the point to the plane.

# Examples
```jldoctest
using Pebsi.Geometry:ptface_mindist
pt = [1,0,0]
face = [0 0 0 0; -1 1 1 -1; -1 -1 1 1]
ptface_mindist(pt,face)
# output
1.0
```
"""
function ptface_mindist(pt::AbstractVector{<:Real},
    face::AbstractMatrix{<:Real})::Real
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
    d = dot(pt - face[:,1],n)
    ppt = pt - d*n
    npts = size(face,2)
    if point_in_polygon(ppt,face)
        abs(d)
    else
        minimum([lineseg₋pt_dist(pt,face[:,[i,mod1(i+1,npts)]]) for i=1:npts])
    end
end

corner_indices = [1,3,6]

@doc """
    affine_trans(pts)

Calculate the affine transformation that maps points to the xy-plane.

# Arguments
- `pts::AbstractMatrix{<:Real}`: Cartesian points as the columns of a matrix.
    The points must all lie on the same plane in 3D.

# Returns
- `::AbstractMatrix{<:Real}`: the affine transformation matrix that operates
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