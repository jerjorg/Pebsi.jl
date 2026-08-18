@doc """
    quadval_vertex(bezcoeffs)

Calculate the value of a 1D quadratic curve at its vertex.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the quadratic polynomial coefficients.

# Returns
- `::Real`: the maximum or minimum value of the quadratic polynomial.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: quadval_vertex
coeffs = [-1, 2, -1]
quadval_vertex(coeffs)
# output
0.5
```
"""
function quadval_vertex(bezcoeffs::AbstractVector{<:Real})::Real
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
end

@doc """
The locations of the quadratic Bezier points at the corners of the triangle in
counterclockwise order for 2D quadratic Bezier points.
"""
corner_indices = [1,3,6]

@doc """
The locations of quadratic Bezier points along each edge of the triangle in
counterclockwise order for 2D quadratic Bezier points.
"""
edge_indices=[[1,2,3],[3,5,6],[6,4,1]]

@doc """
    simplex_intersects(bezpts,atol)

Find the locations where a level curve of a quadratic surface intersects a triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic Bezier surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `intersects::Array`: the intersections organized by edge in a vector. Each element
    of the vector is a matrix where the columns are the Cartesian coordinates of
    theintersections.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simplex_intersects
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -1.0 -2.0 1.0 0.0 2.0]
simplex_intersects(bezpts)
# output
3-element Vector{Array}:
 [-1.0; 0.0;;]
 [0.5; 0.5;;]
 Any[]
```
"""
function simplex_intersects(bezpts::AbstractMatrix{<:Real};
    atol::Real=def_atol)::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = bezcurve_intersects(edge_bezpts[end,:];atol=atol)
        if edge_ints != []
            intersects[i] = reduce(hcat,[edge_bezpts[1:end-1,1] .+ 
                i*(edge_bezpts[1:end-1,end] .- edge_bezpts[1:end-1,1]) for i=edge_ints])
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
    saddlepoint(coeffs;atol)

Calculate the saddle point of a quadratic Bezier surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `::AbstractVector{<:Real}`: the coordinates of the saddle point in barycentric
    coordinates.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: saddlepoint
coeffs = [0.36, -1.64, 0.36, -0.64, -0.64, 0.36]
saddlepoint(coeffs)
# output
3-element Vector{Float64}:
 0.5000000000000001
 0.5000000000000001
 4.163336342344338e-17
```
"""
function saddlepoint(coeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector{<:Real}
    # (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀) = coeffs
    (z₂₀₀, z₁₁₀, z₀₂₀, z₁₀₁, z₀₁₁, z₀₀₂) = coeffs
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
    split_bezsurf₁(bezpts,atol)

Split a Bezier surface once into sub-Bezier surfaces with the Delaunay method.

A triangular mesh is created using a Delaunay tesselation of points at the corners 
of the triangle, the midpoints of the edges, the intersections of the level curves
of the quadratic with the triangle, and the double point of the quadratic surface
if it lies within the triangle. Bezier coefficients for these simplices are calculated.
The goal is to split the quadratic surface into subsurfaces that have no more than
two intersections of the level curves with the edges of the triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces. The sub-surfaces
    reproduce the original Bezier surface.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_bezsurf₁
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 0.0 1.0 -1.0 0.0]
sbezpts = split_bezsurf₁(bezpts)
length(sbezpts)
# output
6
```
"""
function split_bezsurf₁(bezpts::AbstractMatrix{<:Real}; atol::Real=def_atol)::AbstractArray
    spatial = pyimport("scipy.spatial")
    dim = 2; deg = 2; 
    triangle = bezpts[1:end-1,corner_indices]
    if simplex_size(triangle) < def_min_simplex_size
        return [bezpts]
    end
     
    coeffs = bezpts[end,:]; pts = bezpts[1:end-1,:]
    simplex_bpts = sample_simplex(dim,deg)
    intersects = simplex_intersects(bezpts,atol=atol)
    spt = saddlepoint(coeffs)
    allpts = pts
    if insimplex(spt) # Using the default absolute tolerance 1e-12
        allpts = [pts barytocart(spt,triangle)]
    end
    if intersects != [[],[],[]]
        allintersects = reduce(hcat,[i for i=intersects if i!=[]])
        allpts = [allpts allintersects]
    end
    allpts = unique_points(allpts,atol=atol)
    # Had to add box points to prevent collinear triangles.
    xmax,ymax = maximum(bezpts[1:2,:],dims=2)
    xmin,ymin = minimum(bezpts[1:2,:],dims=2)
    xmax += def_mesh_scale*abs(xmax - xmin)
    xmin -= def_mesh_scale*abs(xmax - xmin)
    ymax += def_mesh_scale*abs(ymax - ymin)
    ymin -= def_mesh_scale*abs(ymax - ymin)
    boxpts = [xmin xmax xmax xmin; ymin ymin ymax ymax]
    allpts = [boxpts allpts]
    del = spatial.Delaunay(Matrix(allpts'))
    tri_ind = notbox_simplices(del)
    # For small triangles, all triangles may have a corner at a box corner.
    # In this case, return the original points. Return `bezpts` rather than
    # `pts`: callers expect the coefficients in the last row, and dropping it
    # leaves a matrix one row short that fails downstream in `simplex_size`.
    if length(tri_ind) == 0
        return [bezpts]
    end
    tri_ind = reduce(hcat,tri_ind)
    subtri = [order_vertices!(allpts[:,tri_ind[:,i]]) for i=1:size(tri_ind,2)]
    sub_pts = [barytocart(simplex_bpts,tri) for tri=subtri]
    sub_bpts = [carttobary(pts,triangle) for pts=sub_pts]
    sub_vals = [reduce(hcat, [eval_poly(sub_bpts[j][:,i],coeffs,dim,deg)
        for i=1:6]) for j=1:length(subtri)]
    subtri_coeffs = [getpoly_coeffs(v[:],simplex_bpts,dim,deg) for v=sub_vals]
    sub_bezpts = [[sub_pts[i]; subtri_coeffs[i]'] for i=1:length(subtri_coeffs)]
    sub_bezpts
end

@doc """
    split_bezsurf(bezpts;atol)

Split a Bezier surface into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces. The sub-surfaces
    reproduce the original surface.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_bezsurf
bezpts = [0. 0.5 1 0.5 1 1; 0. 0. 0. 0.5 0.5 1; 1.1 1.2 -1.3 1.4 1.5 1.6]
split_bezsurf(bezpts)
# output
1-element Vector{Matrix{Float64}}:
 [0.0 0.5 … 1.0 1.0; 0.0 0.0 … 0.5 1.0; 1.1 1.2 … 1.5 1.6]
```
*See `split_bezsurf₁` for a more detailed description.
"""
function split_bezsurf(bezpts::AbstractMatrix{<:Real};atol=def_atol)::AbstractArray
    
    intersects = simplex_intersects(bezpts,atol=atol)
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects <= 2
        return [bezpts]
    else
        sub_bezpts = split_bezsurf₁(bezpts)
        sub_intersects = [simplex_intersects(b,atol=atol) for b=sub_bezpts]
        num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
            size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
        while any(num_intersects .> 2)
            # `split_bezsurf₁` returns its input unchanged when the surface
            # cannot be subdivided any further (a degenerate or very small
            # triangle). Re-splitting such a surface makes no progress, so track
            # whether any split succeeded and stop when none did — otherwise
            # this loop never terminates.
            split_occurred = false
            for i = length(num_intersects):-1:1
                if num_intersects[i] <= 2 continue end
                subs = split_bezsurf₁(sub_bezpts[i])
                if length(subs) == 1 continue end
                split_occurred = true
                append!(sub_bezpts,subs)
                deleteat!(sub_bezpts,i)
                sub_intersects = [simplex_intersects(b,atol=atol) for b=sub_bezpts]
                num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 :
                    size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
            end
            if !split_occurred break end
        end
    end
    sub_bezpts
end

@doc """
    sub_coeffs(bezpts,subtriangle)

Calculate the coefficients of a quadratic sub-surface of a quadratic triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic triangle.
- `subtriangle::AbstractMatrix{<:Real}`: the points at the corners of a subtriangle
    as columns of an array.

# Returns
- `::AbstractVector{<:Real}`: the Bezier coefficients that give a subsurface of 
    a quadratic surface. The subsurface has a domain of `subtriangle`.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: sub_coeffs
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.25 -0.25 3.75 -0.25 1.75 1.75]
subtriangle = [-0.5 0.0 -0.6464466094067263; 0.0 1.0 0.35355339059327373]
round.(sub_coeffs(bezpts,subtriangle), digits=10)
# output
6-element Vector{Float64}:
  0.0
  0.25
  1.75
 -0.0732233047
  0.4571067812
 -0.0
```
"""
function sub_coeffs(bezpts::AbstractMatrix{<:Real},
    subtriangle::AbstractMatrix{<:Real})::AbstractVector{<:Real}

    ptsᵢ = carttobary(barytocart(sample_simplex(2,2),subtriangle),bezpts[1:2,corner_indices])
    valsᵢ = eval_poly(ptsᵢ,bezpts[end,:],2,2)
    getpoly_coeffs(valsᵢ,sample_simplex(2,2),2,2)
end

@doc """
    bezcurve_intersects(bezcoeffs;rtol,atol)

Determine where a quadratic curve is equal to zero.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `solutions::AbstractVector`: the locations between [0,1) where the quadratic
    equals 0.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: bezcurve_intersects
coeffs = [0,1,-1]
bezcurve_intersects(coeffs)
# output
2-element Vector{Float64}:
 0.0
 0.6666666666666666
```
"""
function bezcurve_intersects(bezcoeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector
    a,b,c = bezcoeffs
    solutions = solve_quadratic(a - 2*b + c, 2*(b-a), a)
    solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
        && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
    return solutions
end
