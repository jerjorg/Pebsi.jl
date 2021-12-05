module QuadraticIntegration

using SymmetryReduceBZ.Utilities: unique_points, shoelace, remove_duplicates, get_uniquefacets
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ..Polynomials: eval_poly,getpoly_coeffs,getbez_pts₋wts,eval_bezcurve,
    conicsection, evalpoly1D, get_1Dquad_coeffs, solve_quadratic
using ..EPMs: eval_epm, epm₋model, epm₋model2D
using ..Mesh: get_neighbors, notbox_simplices, get_cvpts, ibz_init₋mesh, 
    get_extmesh, choose_neighbors, choose_neighbors3D, trimesh
using ..Geometry: order_vertices!, simplex_size, insimplex, barytocart,
    carttobary, sample_simplex, lineseg₋pt_dist, mapto_xyplane, ptface_mindist
using ..Defaults

using QHull: chull, Chull
using LinearAlgebra: cross, norm, dot, I, diagm, pinv, det
using Statistics: mean
using Base.Iterators: flatten
using PyCall: PyObject, pyimport
using Distributed: pmap
using FastGaussQuadrature: gausslegendre

export bandstructure, init_bandstructure, quadval_vertex, corner_indices, 
    edge_indices, simplex_intersects, saddlepoint, split_bezsurf₁, 
    split_bezsurf, analytic_area, analytic_volume, sub₋coeffs,
    two₋intersects_area₋volume, quad_area₋volume, get_intercoeffs, calc_fl,
    calc_flbe!, refine_mesh!, get_tolerances, quadratic_method, truebe, 
    bezcurve_intersects, getdomain, analytic_area1D, simpson, simpson2D, 
    linept_dist, tetface_areas, simpson3D, quadslice_tanpt, containment_percentage,
    stop_refinement!, calc_fabe

@doc """
    bandstructure

A container for all variables related to the band structure.

# Arguments
- `init_msize::Integer`: the initial size of the mesh over the IBZ. The number
    of points is approximately proportional to init_msize^2/2.
- `num_near_neigh::Integer`: the number of nearest neighbors to consider when 
    calculating interval coefficients. For example, a value of 1 will include 
    neighbors that are a distance of 1 away or are connected by an edge with the
    corners of the simplex.
- `num_neighbors::Integer`: the minimum number of neighbors to include in the 
    calculation of interval coefficients. 
- `fermiarea_eps::Real`: a tolerance used during the bisection algorithm that 
    determines how close the midpoints of the Fermi area interval is to the true
    Fermi area.
- `target_accuracy::Real`: the accuracy desired for the band energy at the end 
    of the computation.
- `fermilevel_method::Integer`: the root-finding method for computing the Fermi 
    level. 1- bisection, 2-Chandrupatla.
- `refine_method::Integer`: the method of refinement. 1-refine the tile with the
    most error. 2-refine the tiles with too much error while taking into account 
    the sizes of the simplices.
- `sample_method::Integer`: the method of sampling a tile with too much error.
    1-add a single point at the center of the triangle. 2-add points at the 
    midpoints of all edges.
- `neighbor_method::Integer`: the method for selecting neighboring points in the 
    calculion of interval coefficients.
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.
- `mesh::PyObject`: a Delaunay triangulation of points over the IBZ.
- `simplicesᵢ::Vector{Vector{Integer}}`: the indices of points at the corners of the 
    tile for all tiles in the triangulation.
- `ext_mesh::PyObject`: a Delaunay triangulation of points within and around the 
    IBZ. The number of points outside is determined by `num_near_neigh`.
- `sym₋unique::AbstractVector{<:Integer}`: the indices of symmetrically unique points
    in the mesh.
- `eigenvals::AbstractMatrix{<:Real}`: the eigenvalues at each of the points unique
    by symmetry.
- `fatten::Real` a variable that scales the size of the interval coefficients.
- `mesh_intcoeffs::Vector{Vector{Matrix{Float64}}}`:the interval Bezier 
    coefficients for all tiles and sheets.
- `mesh_bezcoeffs::Vector{Vector{Vector{Float64}}}`: the least-squares Bezier
    coefficients for all tiles and sheets.
- `fermiarea_interval::AbstractVector{<:Real}`: the Fermi area interval. 
- `fermilevel_interval::AbstractVector{<:Real}`: the Fermi level interval. 
- `bandenergy_interval::AbstractVector{<:Real}`: the band energy interval.
- `fermilevel::Real`: the true Fermi level.
- `bandenergy::Real`: the true band energy.
- `sigma_bandenergy::Vector{<:Real}`: the sigma band energy (the energy of the 
    approximate sheets that are completely below the approximate Fermi level) 
    for each simplex in the mesh.
- `partial_bandenergy::Vector{<:Real}`: the partial band energy (the energy of 
    the approximate sheets that are below and above the approximate Fermi level)
    for each simplex in the mesh.
- `partially_occupied::Vector{Vector{Int64}}`: the sheets that are partially 
    occupied in each tile.
- `bandenergy_errors::Vector{<:Real}`: estimates of the band energy errors in 
    each tile in the mesh.
- `bandenergy_sigma_errors::Vector{<:Real}`: band energy errors from sigma sheets.
- `bandenergy_partial_errors::Vector{<:Real}`: band energy errors from partial
    sheets.
- `fermiarea_errors::AbstractVector{<:Real}`: the Fermi area errors for each tile 
    in the triangulation.
- `weighted::bool`: calculate the interval coefficients using weighted least squares
    if true.
- `constrained::bool`: calculate the interval coefficients with constrained least
    squares if true.
- `stop_criterion::Integer`: determines the criterion used to stop adaptive refinement.
    1: The sum of the estimated band energy errors.
    2: The difference in band energy between two AMR iterations is less the band
      energy accuracy goal.
    3: db/da*Δl is less than the band energy accuracy where db is the derivative of
      the band energy with respect to the Fermi level, da is the derivative of the 
      Fermi area with respect to the Fermi level, and Δl is the uncertainty of the
      Fermi level.
    4 - The number of k-points is greater than the desired number of k-points.
- `target_kpoints::Integer`: the desired number of k-points for the calculation.
    This may be ignored depending on `stop_criterion`.
- `exactfit::Bool`: the polynomial fit goes through the eigenvalues if true.
- `polydegree::Integer`: the degree of the polynomial
"""
mutable struct bandstructure
    init_msize::Integer
    num_near_neigh::Integer
    num_neighbors::Integer
    fermiarea_eps::Real
    target_accuracy::Real
    fermilevel_method::Integer
    refine_method::Integer
    sample_method::Integer
    neighbor_method::Integer
    rtol::Real
    atol::Real
    mesh::PyObject
    simplicesᵢ::Vector{Vector{Integer}}
    ext_mesh::PyObject
    sym₋unique::AbstractVector{<:Integer}
    eigenvals::AbstractMatrix{<:Real}
    fatten::Real
    mesh_intcoeffs::Vector{Vector{Matrix{Float64}}}
    mesh_bezcoeffs::Vector{Vector{Vector{Float64}}}
    fermiarea_interval::AbstractVector{<:Real}
    fermilevel_interval::AbstractVector{<:Real}
    bandenergy_interval::AbstractVector{<:Real}
    fermilevel::Real
    bandenergy::Real
    sigma_bandenergy::Vector{<:Real}
    partial_bandenergy::Vector{<:Real}
    partially_occupied::Vector{Vector{Int64}}
    bandenergy_errors::Vector{<:Real}
    bandenergy_sigma_errors::Vector{<:Real}
    bandenergy_partial_errors::Vector{<:Real}
    fermiarea_errors::Vector{<:Real}
    weighted::Bool
    constrained::Bool
    stop_criterion::Integer
    target_kpoints::Integer
    exactfit::Bool
    polydegree::Integer
end

@doc """
    init_bandstructure(epm,init_msize,num_near_neigh,num_neighbors,fermiarea_eps,
        target_accuracy,fermilevel_method,refine_method,sample_method,neighbor_method,
        fatten,weighted,constrained,stop_criterion,target_kpoints,rtol,atol)

Initialize a band structure container.

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: an empirical pseudopotential.

See the documentation for `bandstructure` for a description of the remaining arguments.

# Returns
- `::bandstructure`: a container for information on the band structure approximation.

# Examples
```jldoctest
import Pebsi.EPMs: m11
import Pebsi.QuadraticIntegration: init_bandstructure,bandstructure
ebs = init_bandstructure(m11)
typeof(ebs)
# output
bandstructure
```
"""
function init_bandstructure(
    epm::Union{epm₋model,epm₋model2D,epm₋model};
    init_msize::Integer=def_init_msize,
    num_near_neigh::Integer=def_num_near_neigh,
    num_neighbors::Union{Nothing,Integer}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Integer=def_fermilevel_method,
    refine_method::Integer=def_refine_method,
    sample_method::Integer=def_sample_method,
    neighbor_method::Integer=def_neighbor_method,
    fatten::Real=def_fatten,
    weighted::Bool=def_weighted,
    constrained::Bool=def_constrained,
    stop_criterion::Integer=def_stop_criterion,
    target_kpoints::Integer=def_target_kpoints,
    exactfit::Bool=false, 
    polydegree::Integer=2,
    rtol::Real=def_rtol,
    atol::Real=def_atol)

    dim = size(epm.recip_latvecs,1)
    mesh = ibz_init₋mesh(epm.ibz,init_msize;rtol=rtol,atol=atol)
    if exactfit
        ext_mesh = mesh; num_neighbors = 0
        simplicesᵢ = notbox_simplices(mesh)
        simplices = [Array(mesh.points[s,:]') for s=simplicesᵢ]
        bspts = sample_simplex(dim,polydegree)
        unique_pts=[barytocart(bspts,s) for s=simplices]
        eigenvals = [eval_epm(p,epm,rtol=rtol,atol=atol) for p=unique_pts]
        mesh_bezcoeffs = [[getpoly_coeffs(eigenvals[i][j,:],bspts,dim,polydegree) 
            for j=1:epm.sheets] for i=1:length(simplicesᵢ)]
        # Lazy, inefficient code. The number of unique points is off by a small amount
        # if the IBZ has symmetrically equivalent boundaries.
        unique_pts = unique_points(reduce(hcat,unique_pts))
        eigenvals = eval_epm(unique_pts,epm,rtol=rtol,atol=atol)
        if polydegree == 1
            if dim == 2
            mesh_bezcoeffs = [[[v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,(v[2]+v[3])/2,v[3]] for v=c] 
                for c=mesh_bezcoeffs]
            else
                mesh_bezcoeffs = [[[v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,(v[2]+v[3])/2,
                    v[3],(v[1]+v[4])/2,(v[2]+v[4])/2,(v[3]+v[4])/2,v[4]] for v=c] 
                    for c=mesh_bezcoeffs]
            end
        end
        mesh_intcoeffs = [[Matrix([mesh_bezcoeffs[i][j] mesh_bezcoeffs[i][j]]') 
            for j=1:epm.sheets] for i=1:length(simplices)]
        sym₋unique = [zeros(Int,2^dim); collect(1:length(simplices)*(2^dim+2))]
        eigenvals = [zeros(epm.sheets,2^dim) eigenvals]
    else
        mesh,ext_mesh,sym₋unique = get_extmesh(epm.ibz,mesh,epm.pointgroup,
            epm.recip_latvecs,num_near_neigh; rtol=rtol,atol=atol)
        simplicesᵢ = notbox_simplices(mesh)
        uniqueᵢ = sort(unique(sym₋unique))[2:end]
        estart = if dim == 2 4 else 8 end
        eigenvals = zeros(Float64,epm.sheets,estart+length(uniqueᵢ))
        for i=uniqueᵢ
            eigenvals[:,i] = eval_epm(mesh.points[i,:], epm, rtol=rtol, atol=atol)
        end
        if num_neighbors == nothing
            num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
        end
        coeffs = [get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,
            simplicesᵢ,fatten,num_near_neigh,epm=epm,neighbor_method=neighbor_method,
            num_neighbors=num_neighbors, weighted=weighted, constrained=constrained) 
            for index=1:length(simplicesᵢ)]
        mesh_intcoeffs = [coeffs[index][1] for index=1:length(simplicesᵢ)]
        mesh_bezcoeffs = [coeffs[index][2] for index=1:length(simplicesᵢ)]
    end
    
    partially_occupied = [zeros(Int,epm.sheets) for _=1:length(simplicesᵢ)]
    bandenergy_errors = zeros(length(simplicesᵢ))
    bandenergy_sigma_errors = zeros(length(simplicesᵢ))
    bandenergy_partial_errors = zeros(length(simplicesᵢ))
    fermiarea_errors = zeros(length(simplicesᵢ))
    sigma_bandenergy = zeros(length(simplicesᵢ))
    partial_bandenergy = zeros(length(simplicesᵢ)) 
    if exactfit
        fermilevel_interval = [minimum(eigenvals), maximum(eigenvals)]
    else
        fermilevel_interval=[0,0]
    end
    fermiarea_interval=[0,0]; bandenergy_interval=[0,0]; fermilevel=0; bandenergy=0
        
    bandstructure(
        init_msize,
        num_near_neigh,
        num_neighbors,
        fermiarea_eps,
        target_accuracy,
        fermilevel_method,
        refine_method,
        sample_method,
        neighbor_method,
        rtol,
        atol,
        
        mesh,
        simplicesᵢ,
        ext_mesh,
        sym₋unique,
        eigenvals,
        fatten,
        mesh_intcoeffs,
        mesh_bezcoeffs,
        fermiarea_interval,
        fermilevel_interval,
        bandenergy_interval,
        fermilevel,
        bandenergy,
        sigma_bandenergy,
        partial_bandenergy,
        partially_occupied,
        bandenergy_errors,
        bandenergy_sigma_errors,
        bandenergy_partial_errors,
        fermiarea_errors,

        weighted,
        constrained,
        stop_criterion,
        target_kpoints,
        exactfit,
        polydegree)
end   

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
 [-1.0; 0.0]
 [0.5; 0.5]
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
 4.163336342344338e-17
 0.5000000000000001
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
    # In this case, return the original points.
    if length(tri_ind) == 0
        return [pts]
    end
    tri_ind = reduce(hcat,tri_ind)
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
            for i = length(num_intersects):-1:1
                if num_intersects[i] <= 2 continue end
                append!(sub_bezpts,split_bezsurf₁(sub_bezpts[i]))
                deleteat!(sub_bezpts,i)
                sub_intersects = [simplex_intersects(b,atol=atol) for b=sub_bezpts]
                num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
                    size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
            end
        end
    end
    sub_bezpts
end

@doc """
    analytic_area(w::Real)

Calculate the area within a triangle and a canonical, rational Bezier curve.

The canonical triangle has corners at [-1,0], [1,0], and [0,1]. The weights of the
rational Bezier curve at corners [-1,0] and [1,0] are 1, so the only free parameter
is the weight of the middle Bezier point at [0,1]. See the notebook 
`analytic-expressions-derivation.nb` for a derivation of the analytic expression
and the Taylor expansion of the approximation.

# Arguments
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic Bezier
    curve.

# Returns
- `::Real`: the area within the triangle and Bezier curve.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: analytic_area
w = 1.0
analytic_area(w)
# output
0.6666666666666666
```
"""
function analytic_area(w::Real)::Real
    
    # Use the Taylor expansion of the analytic expression if the weight is close to 1.
    if isapprox(w,1,atol=def_taylor_exp_tol)
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
    analytic_volume(coeffs,w)

Calculate the volume within a canonical triangle and Bezier curve of a quadratic surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic surface.
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic Bezier curve.

# Returns
- `::Real`: the volume of the quadratic surface within the region bounded by the 
    triangle and the rational Bezier curve.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: analytic_volume
coeffs = [0.2,0.2,0.3,-0.3,0.4,-0.4]
w = 0.3
analytic_volume(coeffs,w)
# output
-0.029814783582691722
```
"""
function analytic_volume(coeffs::AbstractVector{<:Real},w::Real)::Real
    
    (c₅,c₃,c₀,c₄,c₁,c₂) = coeffs
    d = c₀+c₁+c₂+c₃+c₄+c₅
    # Use the Taylor expansion of the analytic solution if the weight is close to 1.
    if isapprox(w,1,atol=def_taylor_exp_tol)
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
        sign(w)real((w*(a*b*w*(-32*c₁+33*c₂-32*c₃+46*c₄+33*c₅-2*(-26*c₀+18*c₁+13*c₂+18*c₃+12*c₄+13*c₅)*w^2+
            8*d*w^4)+6*(5*c₂+6*c₄+5*c₅+4*(c₀-5*(c₁+c₃)+c₄)*w^2+16*c₀*w^4)*atan(b/a)))/(6*8*a*b*(-1+w^2)^3))
    end
end

@doc """
    sub₋coeffs(bezpts,subtriangle)

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
import Pebsi.QuadraticIntegration: sub₋coeffs
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.25 -0.25 3.75 -0.25 1.75 1.75]
subtriangle = [-0.5 0.0 -0.6464466094067263; 0.0 1.0 0.35355339059327373]
sub₋coeffs(bezpts,subtriangle)
# output
6-element Vector{Float64}:
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
    two₋intersects_area₋volume(bezpts,quantity;atol)

Calculate the area or volume within a quadratic curve and triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of a quadratic surface.
- `quantity::String`: the quantity to compute ("area" or "volume").
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `areaₒᵣvolume::Real`: the area within the curve and triangle or the volume below
    the surface within the curve and triangle. The area is on the side of the curve
    where the surface is less than zero.

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
    quantity::String; atol::Real=def_atol)::Real
     
    # Calculate the bezier curve and weights, make sure the curve passes through
    # the triangle
    triangle = bezpts[1:end-1,corner_indices]
    coeffs = bezpts[end,:]
    intersects = simplex_intersects(bezpts,atol=atol)
    # No intersections
    if intersects == [[],[],[]]
        # Case where the sheet is completely above or below 0.    
        if all(coeffs .< 0) && !any(isapprox.(coeffs,0,atol=atol))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
            return areaₒᵣvolume
        end
        if all(coeffs .> 0) && !any(isapprox.(coeffs,0,atol=atol))
            areaₒᵣvolume = 0
            return areaₒᵣvolume
        end
    end

    bezptsᵣ = []
    if intersects != [[],[],[]]
        all_intersects = reduce(hcat,[i for i=intersects if i != []])
        if size(all_intersects,2) != 2
            error("Can only calculate the area or volume when the curve intersects 
                the triangle at two points or doesn't intersect the triangle.")
        end
        p₀ = all_intersects[:,1]
        p₂ = all_intersects[:,2]
        (bezptsᵣ,bezwtsᵣ) = getbez_pts₋wts(bezpts,p₀,p₂,atol=atol)
        ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
        # Make sure the weight of the middle Bezier point has the correct sign.
        if !insimplex(carttobary(ptᵣ,triangle))
            bezwtsᵣ[2] *= -1
            ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
            if !insimplex(carttobary(ptᵣ,triangle))
                intersects = [[],[],[]]
            end
        else
            # Remove intersections if the mipoint of the Bezier curve is on an edge.
            on_edge = any(isapprox.([lineseg₋pt_dist(ptᵣ,triangle[:,i],atol=atol) 
                for i=[[1,2],[2,3],[3,1]]],0,atol=atol))
            if on_edge intersects = [[],[],[]] end
        end
    end

    # If the tangent lines are close to parallel, the middle Bezier point of the
    # curve will be very far away, which introduces numerical errors. We handle
    # this by splitting the surface up and recalculating.
    cstype = conicsection(coeffs) # using the default tolerance of 1e-12    
    linear = any(cstype .== ["line","rectangular hyperbola","parallel lines"])
    split = false
    if bezptsᵣ != []
        if maximum(abs.(bezptsᵣ)) > def_rational_bezpt_dist 
            split = true
        end
    end

    # Split the triangle if the saddle point is within the triangle but not on a
    # corner and the conic section is linear or degenerate.
    saddle = saddlepoint(coeffs,atol=atol)
    if insimplex(saddle) && !any([isapprox(saddle,x,atol=atol) for x=[[1,0,0],[0,1,0],[0,0,1]]])
        split = true
    end

    if split
        bezptsᵤ = [split_bezsurf(b,atol=atol) for b=split_bezsurf₁(bezpts)] |> flatten |> collect
        return sum([two₋intersects_area₋volume(b,quantity,atol=atol) for b=bezptsᵤ])
    end

    # No intersections, no island, and the coefficients are less or greater than 0.
    if intersects == [[],[],[]]
        v = eval_poly([1/3,1/3,1/3],coeffs,2,2)
        if v < 0 || isapprox(v,0,atol=atol)
            below = true
        else
            below = false
        end

        if below
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

        # If two intersections on the same edge, use a point that is the average of 
        # the intersections and the midpoint of the Bezier curve.
        avept = mean(all_intersects,dims=2) |> vec
        avept = mean([avept ptᵣ],dims=2) |> vec
    elseif length(edgesᵢ) ==2
        corner = [3,1,2][setdiff([1,2,3],edgesᵢ)[1]]
        # If intersections on different edges, use a point that is the average of
        # the intersections and the corner.
        avept = mean([ptᵣ triangle[:,corner]],dims=2) |> vec
    else
        error("The curve may only intersect at most two edges.")
    end
    avept = carttobary(avept,triangle)
    if (eval_poly(avept,coeffs,2,2) < 0 || 
        isapprox(eval_poly(avept,coeffs,2,2), 0, atol=atol))
        below₀ = true
    else
        below₀ = false
    end 

    simplex_bpts = sample_simplex(2,2)
    triangleₑ = order_vertices!([all_intersects triangle[:,corner]])
    if quantity == "area"
        # curve area or volume
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_area(bezwtsᵣ[2])
    elseif quantity == "volume"
        coeffsᵣ = sub₋coeffs(bezpts,bezptsᵣ)
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_volume(coeffsᵣ,bezwtsᵣ[2])
    else
        throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
    end

    # Get the sign of the area correct (accounts for the curvature of the curve).
    inside = false
    # Get exception when corners of triangleₑ all lie on a straight line.
    try
        inside = insimplex(carttobary(ptᵣ,triangleₑ))
    catch SingularException
        nothing
    end

    if length(edgesᵢ) == 2 && inside
        areaₒᵣvolumeᵣ *= -1
    end
    
    if quantity == "area"
        areaₒᵣvolume =  areaₒᵣvolumeᵣ + simplex_size(triangleₑ)
        if !below₀
            areaₒᵣvolume = simplex_size(triangle) - areaₒᵣvolume
        end
    else # quantity == "volume"
        coeffsₑ = sub₋coeffs(bezpts,triangleₑ)
        areaₒᵣvolume = mean(coeffsₑ)*simplex_size(triangleₑ) + areaₒᵣvolumeᵣ
        if !below₀
            areaₒᵣvolume = simplex_size(triangle)*mean(coeffs) - areaₒᵣvolume
        end
    end

    areaₒᵣvolume
end

@doc """
    quad_area₋volume(bezpts,quantity;atol)

Calculate the area of the shadow or the volume beneath a quadratic.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface in
    columns of a matrix.
- `quantity::String`: the quantity to calculate ("area" or "volume").
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `::Real`: the area of the shadow of a quadratic polynomial within a triangle
    and below the plane `z=0` or the volume of the quadratic polynomial under the 
    same constraints.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: quad_area₋volume
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2/3 -4/3 2/3 -2/3 -2/3 0]
quad_area₋volume(bezpts,"area") ≈ 0.8696051011068969
# output
true
```
"""
function quad_area₋volume(bezpts::AbstractMatrix{<:Real},
        quantity::String;atol::Real=def_atol)::Real

    # Modifications to make when working in 3D.
    if size(bezpts,1) == 4
        bezpts = [mapto_xyplane(bezpts[1:3,:]); bezpts[end,:]']
    end
    sum([two₋intersects_area₋volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])
end

face_ind = [[1,2,3],[3,4,1],[4,1,2],[2,3,4]]

@doc """
    get_intercoeffs(index, mesh, ext_mesh, sym₋unique, eigenvals, simplicesᵢ,
        fatten, num_near_neigh; sigma, epm, neighbor_method, num_neighbors, weighted,
        constrained, atol)

Calculate the interval Bezier points for all sheets.

# Arguments
- `index::Integer`: the index of the simplex in `simplicesᵢ`.
- `mesh::PyObject`: a triangulation of the irreducible Brillouin zone.
- `ext_mesh::PyObject`: a triangulation of the region within and around the IBZ.
- `sym₋unique::AbstractVector{<:Real}`: the index of the eigenvalues for each point
    in the `mesh`.
- `eigenvals::AbstractMatrix{<:Real}`: a matrix of eigenvalues for the symmetrically
    distinc points as columns of a matrix.
- `simplicesᵢ::AbstractVector`: the simplices of `mesh` that do not include the box points.
- `fatten::Real=def_fatten`: scale the interval coefficients by this amount.
- `num_near_neigh::Integer=def_num_near_neigh`: how many nearest neighbors to include.
- `sigma::Integer=0`: the number of sheets summed and then interpolated, if any.
- `epm::Union{Nothing,epm₋model2D,epm₋model}=nothing`: an empirical pseudopotential.
- `neighbor_method::Integer=def_neighbor_method`: the method for calculating neighbors
    to include in the calculation.
- `num_neighbors::Union{Nothing,Integer}=nothing`: the minimum number of neighbors
    included in the calculation of interval coefficients.
- `weighted::Bool=false`: if true, points are weighted by their minimum distance to
    a boundary of the simplex.
- `constrained::Bool=true`: if true, use constrained least squares.
- `atol::Real=def_atol`: an absolume tolerance.

# Returns
- `inter_bezpts::Vector{Matrix{Float64}}`: the interval Bezier points for each sheet.
- `bezcoeffs::Vector{Vector{Float64}}`: the Bezier coefficients from the least-squares
    fit for each sheet.

# Examples
```jldoctest
import Pebsi.EPMs: m2ibz,m2pointgroup,m2recip_latvecs,m2rules,m2cutoff,eval_epm
import Pebsi.Mesh: ibz_init₋mesh, get_extmesh, notbox_simplices
import Pebsi.QuadraticIntegration: get_intercoeffs
n = 10
mesh = ibz_init₋mesh(m2ibz,n)
simplicesᵢ = notbox_simplices(mesh)
num_near_neigh = 2
mesh,ext_mesh,sym₋unique = get_extmesh(m2ibz,mesh,m2pointgroup,m2recip_latvecs,num_near_neigh)
sheets = 7
energy_conv = 1
eigenvals = zeros(sheets,size(mesh.points,1))
for i = sort(unique(sym₋unique))[2:end]
    eigenvals[:,i] = eval_epm(mesh.points[i,:],m2recip_latvecs,m2rules,m2cutoff,sheets,energy_conv)
end
index = 1
intercoeffs,bezcoeffs = get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,simplicesᵢ)
length(bezcoeffs)
# output
7
```
"""
function get_intercoeffs(index::Integer, mesh::PyObject, ext_mesh::PyObject,
    sym₋unique::AbstractVector{<:Real}, eigenvals::AbstractMatrix{<:Real}, 
    simplicesᵢ::AbstractVector, fatten::Real=def_fatten, 
    num_near_neigh::Integer=def_num_near_neigh; sigma::Real=0, 
    epm::Union{Nothing,epm₋model2D,epm₋model}=nothing,
    neighbor_method::Integer=def_neighbor_method, 
    num_neighbors::Union{Nothing,Integer}=nothing,
    weighted::Bool=false,constrained::Bool=true,atol::Real=def_atol)
    
    simplexᵢ = simplicesᵢ[index]
    simplex = Matrix(mesh.points[simplexᵢ,:]')
    dim = size(simplex,2)-1 
    neighborsᵢ = reduce(vcat,[get_neighbors(s,ext_mesh,num_near_neigh) for s=simplexᵢ]) |> unique
    neighborsᵢ = filter(x -> !(x in simplexᵢ),neighborsᵢ)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end

    if length(neighborsᵢ) < num_neighbors num_neighbors = length(neighborsᵢ) end

    # Select neighbors that are closest to the triangle.
    if neighbor_method == 1
        neighbors = ext_mesh.points[neighborsᵢ,:]'
        dist = [minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:dim+1]) for i=neighborsᵢ]
        neighborsᵢ = neighborsᵢ[sortperm(dist)][1:num_neighbors]
    # Select neighbors that surround the triangle and are close to the triangle.
    elseif neighbor_method == 2
        neighbors = Matrix(ext_mesh.points[neighborsᵢ,:]')
        if dim == 2
            neighborsᵢ = choose_neighbors(simplex,neighborsᵢ,neighbors; num_neighbors=num_neighbors)
        else
            neighborsᵢ = choose_neighbors3D(simplex,neighborsᵢ,neighbors; num_neighbors=num_neighbors)
        end
    # Neighbors are taken from a uniform grid within the triangle.
    elseif neighbor_method == 3
        neighborsᵢ = []
        if epm == nothing
            error("Must provide an EPM when computing neighbors within the triangle.")
        end
    else
        error("Only 1, 2, and 3 are valid values of the flag for the method of selecting neighbors.")
    end

    if neighbor_method == 3
        n = def_inside_neighbors_divs # Number of points for the uniform sampling of the triangle
        b = sample_simplex(dim,n)
        b = b[:,setdiff(1:size(b,2),[1,n+1,size(b,2)])]
        eigvals = eval_epm(barytocart(b,simplex),epm)
    else
        b = carttobary(ext_mesh.points[neighborsᵢ,:]',simplex)
    end

    # Constrained least-squares
    if constrained
        if dim == 2
            M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3]],b,dims=1)'
        else
            M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3],x[1]*x[4],x[2]*x[4],x[3]*x[4]],b,dims=1)'
        end
    # Unconstrained least-squares
    else
        if dim == 2
            b = [[1 0 0; 0 1 0; 0 0 1] b]
            M = mapslices(x->[x[1]^2,2*x[1]*x[2],x[2]^2,2*x[1]*x[3],2*x[2]*x[3],x[3]^2],b,dims=1)'
        else
            b = [[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] b]
            M = mapslices(x->[x[1]^2,2*x[1]*x[2],x[2]^2,2*x[1]*x[3],2*x[2]*x[3],x[3]^2,
                2*x[1]*x[4],2*x[2]*x[4],2*x[3]*x[4],x[4]^2],b,dims=1)' 
        end
    end

    # Weighted least squares
    if weighted
        if dim == 2
            # Minimum distance from the edges of the triangle
            W = diagm([minimum([lineseg₋pt_dist(ext_mesh.points[i,:],simplex[:,s]) for s=[[1,2],[2,3],[3,1]]])
                for i=neighborsᵢ])
        else
            # Minimum distance from the faces of the tetrahedron 
            W = diagm([minimum([ptface_mindist(ext_mesh.points[i,:],simplex[:,s]) for s=face_ind])
                for i=neighborsᵢ])
        end
    else
        W = I
    end
 
    nterms = sum([i for i=1:dim+1])
    if sigma == 0
        bezcoeffs = [zeros(nterms) for i=1:size(eigenvals,1)]
        inter_bezcoeffs = [zeros(dim,6) for i=1:size(eigenvals,1)]
    else
        bezcoeffs = [zeros(nterms) for i=1:1]
        inter_bezcoeffs = [zeros(dim,6) for i=1:1]
    end

    for sheet = 1:size(eigenvals,1)
        if sigma == 0
            if neighbor_method != 3
                fᵢ = eigenvals[sheet,sym₋unique[neighborsᵢ]]
            else
                fᵢ = eigvals[sheet,:]
            end
            q = eigenvals[sheet,sym₋unique[simplexᵢ]]
        else
            if neighbor_method != 3
                fᵢ = [sum(eigenvals[1:sigma,sym₋unique[neighborsᵢ]],dims=1)...]
            else
                fᵢ = [sum(eigvals[1:sigma,:],dims=1)...]
            end
            q = [sum(eigenvals[1:sigma,sym₋unique[simplexᵢ]],dims=1)...]
        end

        if constrained
            Z = fᵢ - (b.^2)'*q
        else
            fᵢ = [q; fᵢ];
            Z = fᵢ
        end

        if weighted
            MWM = M'*W*M
            if isapprox(det(MWM),0,atol=atol)
                c = pinv(MWM)*M'*W*Z
            else
                c = inv(MWM)*M'*W*Z
            end
        else
            # c = pinv(M)*Z        
            c = M\Z
        end
        if dim == 2
            c1,c2,c3 = c
            q1,q2,q3 = q
        else
            c1,c2,c3,c4,c5,c6 = c
            q1,q2,q3,q4 = q
        end

        if dim == 2        
            scoeffs = [q1,c1,q2,c2,c3,q3]
        else
            scoeffs = [q1,c1,q2,c2,c3,q3,c4,c5,c6,q4]
        end

        if sigma == 0
            if constrained
                bezcoeffs[sheet] = scoeffs
            else
                bezcoeffs[sheet] = c
            end
        else
            if constrained
                bezcoeffs[1] = scoeffs
            else
                bezcoeffs[1] = c
            end
        end
        if constrained
            qᵢ = [eval_poly(b[:,i],scoeffs,dim,2) for i=1:size(b,2)]
        else
            qᵢ = [eval_poly(b[:,i],c,dim,2) for i=1:size(b,2)]
        end
        δᵢ = fᵢ - qᵢ; 
        ϵ = Matrix(reduce(hcat,[(1/dot(M[i,:],M[i,:])*δᵢ[i])*M[i,:] for i=1:length(δᵢ)])')
        ϵ = [minimum(ϵ,dims=1); maximum(ϵ,dims=1)].*fatten 
        intercoeffs = [c';c'] .+ ϵ
        if constrained
            if dim == 2
                c1,c2,c3 = [intercoeffs[:,i] for i=1:3]
                intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3]])
            else
                c1,c2,c3,c4,c5,c6 = [intercoeffs[:,i] for i=1:6]
                intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3],c4,c5,c6,[q4,q4]])
            end
        end
        if sigma == 0
            inter_bezcoeffs[sheet] = intercoeffs
        else
            inter_bezcoeffs[1] = intercoeffs
            break
        end
    end

    Vector{Matrix{Float64}}(inter_bezcoeffs),Vector{Vector{Float64}}(bezcoeffs)
end

@doc """
    calc_fl(epm,ebs;num_slices,window,ctype,fermi_area,test)

    Calculate the Fermi level for a representation of the band structure.

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: an empirical pseudopotential 
- `ebs::bandstructure`: a `bandstructure` data structure.
- `num_slices::Int=10`: the number of slices for integration in 3D.
- `window::Union{Nothing,Vector{<:Real}}=ebs.fermilevel_interval`: an energy window
    that bounds the Fermi level.
- `ctype="mean"`: determines the coefficients that are used to compute the Fermi level.
   Options include: "mean"- use the coefficients obtained from the least-squares fit,
   "min"- use the lower coefficients of the interval coefficients and "max"- use 
   the upper coefficients of the interval coefficients.
- `fermi_area::Real=epm.fermiarea/length(epm.pointgroup)`: the sum of the areas of
    the shadows of the sheets.
- `test::Bool`: used to test the efficiency of the root-finding algorithm.

# Returns
- `E::Real`: the Fermi level for the quadratic approximation of the band structure.

# Examples
```jldoctest
using Pebsi.EPMs: m21
using Pebsi.QuadraticIntegration: init_bandstructure, calc_fl
epm = m21
ebs = init_bandstructure(epm);
calc_fl(epm,ebs)
# output
0.06335261485115436
```
"""
function calc_fl(epm::Union{epm₋model,epm₋model2D},ebs::bandstructure; 
    num_slices::Int=def_num_slices, window::Vector{<:Real}=ebs.fermilevel_interval, 
        ctype::String="mean", fermi_area::Real=epm.fermiarea/length(epm.pointgroup),
        test::Bool=false)

    if !(ctype in ["min","max","mean"])
        error("Invalid ctype.")
    end    
    dim = size(epm.recip_latvecs,1)
    # simplex_bpts = sample_simplex(dim,2)
    # simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    
    ibz_area = epm.ibz.volume
    maxsheet = round(Int,epm.electrons/2) + 2

    iters = 0    
    estart = if dim == 2 5 else 9 end # Don't consider points at the corners of the box
    if window == nothing || window == [0,0]
        E₁ = minimum(ebs.eigenvals[1,estart:end])
        E₂ = maximum(ebs.eigenvals[maxsheet,estart:end])
    else
        E₁,E₂ = window
    end

    # Make sure the window contains the approx. Fermi level.
    dE = 2*abs(E₂ - E₁)
    f₁ = calc_fabe(ebs, quantity="area", ctype="mean",fl=E₁, num_slices=num_slices) - fermi_area
    iters₁ = 1
    while f₁ > 0
        iters₁ += 1; E₁ -= dE; dE *= 2
        if iters₁ > def_fl_max_iters || dE == 0
            E₁ = minimum(ebs.eigenvals[1,:5:end])
        end
        f₁ = calc_fabe(ebs, quantity="area", ctype="mean", fl=E₁, num_slices=num_slices) - fermi_area
    end

    dE = 2*abs(E₂ - E₁)
    f₂ = calc_fabe(ebs, quantity="area", ctype="mean", fl=E₂, num_slices=num_slices) - fermi_area     
    iters₂ = 1
    while f₂ < 0
        iters₂ += 1; E₂ += dE; dE *= 2
        if iters₂ > def_fl_max_iters || dE == 0
            E₂ = maximum(ebs.eigenvals[maxsheet,5:end])
        end
        f₂ = calc_fabe(ebs, quantity="area", ctype="mean", fl=E₂,num_slices=num_slices) - fermi_area
    end
    E = (E₁ + E₂)/2
    f₃,E₃,iters,f,t = 0,0,0,1e9,0
    ϵ = def_chandrupatla_tol
    while abs(f) > ebs.fermiarea_eps
        iters += 1
        if iters > def_fl_max_iters
            @warn "Failed to converge the Fermi area to within the provided tolerance of $(ebs.fermiarea_eps) after $(def_fl_max_iters) iterations. Fermi area converged within $(f)."
                break
        end
        f = calc_fabe(ebs, quantity="area", ctype=ctype, fl=E, num_slices=num_slices) - fermi_area

        if sign(f) != sign(f₁)
            E₃ = E₂
            f₃ = f₂
            E₂ = E₁
            f₂ = f₁
            E₁ = E
            f₁ = f
        else
            E₃ = E₁
            f₃ = f₁
            E₁ = E
            f₁ = f
        end

        # Bisection method
        if ebs.fermilevel_method == 1
            t = 0.5
        # Chandrupatla method
        elseif ebs.fermilevel_method == 2            
            ϕ₁ = (f₁ - f₂)/(f₃ - f₂)
            ξ₁ = (E₁ - E₂)/(E₃ - E₂)
            if 1 - √(1 - ξ₁) < ϕ₁ && ϕ₁ < √ξ₁
                α = (E₃ - E₁)/(E₂ - E₁)
                t = (f₁/(f₁ - f₂))*(f₃/(f₃ - f₂)) - α*(f₁/(f₃ - f₁))*(f₂/(f₂ - f₃))
            else
                t = 0.5
            end
            if t < ϵ
                t = ϵ
            elseif t > 1-ϵ
                t = 1-ϵ
            end
        else
            ArgumentError("The method for calculating the Fermi is either 1 or 2.")
        end
        E = E₁ + t*(E₂ - E₁)
    end
    if test 
        (iters₁ + iters₂ + iters, E)
    else
        E
    end
end

@doc """
    calc_flbe!(epm,ebs;num_slices)

Calculate the Fermi level and band energy for a given rep. of the band struct.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ebs::bandstructure`: the band structure container.
- `num_slices::Integer=10`: the number of slices when integrating in 3D.
- `flerrors::Bool=true`: if true, band energy errors include effects from Fermi 
    level error.

# Returns
- `ebs::bandstructure`: updated values within container for the band energy error,
    Fermi area error, Fermi level interval, Fermi area interval, band energy
    interval, and the partially occupied sheets.

# Examples
```jldoctest
using Pebsi.EPMs: m31
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!
epm = m31
ebs = init_bandstructure(epm);
calc_flbe!(epm,ebs)
ebs.bandenergy
# output
0.013600374450169372
```
"""
function calc_flbe!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure;
    num_slices::Integer=10, flerrors::Bool=true)::bandstructure
     
    # The number of point operators
    npg = length(epm.pointgroup)
    if ebs.exactfit == true
        fl = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="mean", num_slices=num_slices)
        be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        ebs.fermilevel = fl; ebs.bandenergy = 2*npg*(be + fl*epm.fermiarea/npg)
        return ebs
    end
    
    dim = size(epm.recip_latvecs,1)
    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(dim,2) 
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The areas of the triangles in the mesh
    simplex_sizes = [simplex_size(s) for s=simplices]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]    
    # The number of simplices
    ns = length(ebs.simplicesᵢ)
    # The "true" Fermi level for the given approximation of the band structure
    fl = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="mean", num_slices=num_slices)
    # The larger Fermi level computed with the upper limit of approximation intervals
    fl₁ = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="max", num_slices=num_slices)
    # The smaller Fermi level computed with the lower limit of approximation intervals
    fl₀ = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="min", num_slices=num_slices)

    # The "true" Fermi area for each quadratic triangle (triangle and sheet) for the
    # given approximation of the bandstructure
    mesh_fa = calc_fabe(ebs, quantity="area", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=false)
     
    # The smaller Fermi area for each quadratic triangle using the upper limit of the approximation
    # intervals with the lower limit of the Fermi level interval
    if !flerrors
        mesh_fa₁ = calc_fabe(ebs, quantity="area", ctype="min", fl=fl, num_slices=num_slices,
            sum_fabe=false)
        mesh_fa₀ = calc_fabe(ebs, quantity="area", ctype="max", fl=fl, num_slices=num_slices,
            sum_fabe=false)
    # The larger Fermi area for each quadratic triangle using the lower limit of the approximation
    # intervals with the upper limit of the Fermi level interval
    else
        mesh_fa₁ = calc_fabe(ebs, quantity="area", ctype="min", fl=fl₁, num_slices=num_slices,
            sum_fabe=false)
        mesh_fa₀ = calc_fabe(ebs, quantity="area", ctype="max", fl=fl₀, num_slices=num_slices,
            sum_fabe=false)
    end

    # The smaller and larger Fermi areas or the limits of the Fermi area interval
    fa₀,fa₁ = sum(sum(mesh_fa₀)),sum(sum(mesh_fa₁))
    # The "true" band energy for the given approximation of the band structure
    be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=true)

    # The Fermi area errors for each quadratic triangle (triangle and sheet) for the
    # given band structure approximation
    mesh_fa₋errs = mesh_fa₁ .- mesh_fa₀
     
    # Determine which triangles and sheets are partially occupied by comparing
    # the Fermi area (shadows of the sheets) for each quadratic triangle to zero
    # and the area of the triangle.
    partial_occ = [[(
        if (isapprox(mesh_fa₁[tri][sheet],0,atol=ebs.atol) &&
            isapprox(mesh_fa₀[tri][sheet],0,atol=ebs.atol))
            2
        elseif (isapprox(mesh_fa₁[tri][sheet],simplex_sizes[tri],atol=ebs.atol) &&
            isapprox(mesh_fa₀[tri][sheet],simplex_sizes[tri],atol=ebs.atol))
            0
        else
            1
        end
    ) for sheet=1:epm.sheets] for tri = 1:ns]

    # Determine which triangle and sheets are occupied, partially occupied, or 
    # unoccupied for the "true" approximation of the bandstructure (least squares
    # fitting of eigenvalues).
    true_partial_occ = [[(
        if isapprox(mesh_fa[tri][sheet],0,atol=ebs.atol)
            2
        elseif isapprox(mesh_fa[tri][sheet],simplex_sizes[tri],atol=ebs.atol)
            0
        else
            1
        end
    ) for sheet=1:epm.sheets] for tri = 1:ns]
    
    # The largest index of sheets that are completely occupied (integer) for each triangle
    sigmas = [findlast(x->x==0,partial_occ[i]) for i=1:length(partial_occ)]
    # The indices of sheets that are partially occupied (vector) for each triangle
    partials = [findall(x->x==1,partial_occ[i]) for i=1:length(partial_occ)]
     
    true_sigmas = [findlast(x->x==0,true_partial_occ[i]) for i=1:length(true_partial_occ)]
    true_partials = [findall(x->x==1,true_partial_occ[i]) for i=1:length(true_partial_occ)] 
     
    # For each triangle, sum the eigenvalues of all sheets that are completely occupied
    # and calculate Bezier coeffients and intervals coefficients for the "sigma" sheet.
    nterms = sum([i for i=1:dim+1])
    sigma_intcoeffs = [
        (if sigmas[i] == nothing
            [[zeros(2,nterms)],[zeros(1,nterms)]]
        else
            get_intercoeffs(i,ebs.mesh,ebs.ext_mesh,ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,sigma=sigmas[i],epm=epm,
            neighbor_method = ebs.neighbor_method)
        end) for i=1:ns]

    # Calculate the "sigma" coefficients for the "true" occupations of the sheets. The true sigma 
    # coefficients and intervals and the regular coefficients and intervals are only different if the
    # occupations are different
    true_sigma_intcoeffs = [
        (if true_sigmas[i] == nothing
            [[zeros(2,nterms)],[zeros(1,nterms)]]
        else
            get_intercoeffs(i,ebs.mesh,ebs.ext_mesh,ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,sigma=true_sigmas[i],epm=epm,
            neighbor_method = ebs.neighbor_method)
        end) for i=1:ns]
    
    # Assign the sigma intervals and coefficients their own variables for both true and regular.
    sigma_intervals = [sigma_intcoeffs[i][1][1] for i=1:length(sigma_intcoeffs)]
    sigma_coeffs = [sigma_intcoeffs[i][2][1] for i=1:length(sigma_intcoeffs)]
    
    true_sigma_intervals = [true_sigma_intcoeffs[i][1][1] for i=1:length(true_sigma_intcoeffs)]
    true_sigma_coeffs = [true_sigma_intcoeffs[i][2][1] for i=1:length(true_sigma_intcoeffs)]

    # Calculate the contribution to the band energy of the sigma sheets in each triangle using the 
    # "true" coefficients and the "true" occupations.
    sigma_be = [simplex_size(simplices[i])*mean(true_sigma_coeffs[i]) for i=1:length(true_sigma_intcoeffs)]

    # Calculate a lower limit for the sigma contribution to the band energy using the regular 
    # occupations and the lower limit of the sigma interval coefficients.
    sigma_be₀ = [#sigma_intervals[i] == 0 ? 0 : 
        simplex_size(simplices[i])*mean(sigma_intervals[i][1,:]) for i=1:length(sigma_intcoeffs)]
    # Calculate the upper limit of the sigma contribution to the band energy using the regular
    # occupations and the upper limit of the sigma interval coefficients.
    sigma_be₁ = [#sigma_intervals[i] == 0 ? 0 : 
        simplex_size(simplices[i])*mean(sigma_intervals[i][2,:]) for i=1:length(sigma_intcoeffs)]
    
    # The contribution to the band energy error from the sigma sheets in each triangle
    sigma_be_errs = (sigma_be₁ - sigma_be₀)./2 # the average error

    # The "true" contribution to the band energy from the partially occupied sheets using the
    # least-squares coefficients and the true occupations. The leading term takes into account
    # the integral transform.
    partial_be = fl*mesh_fa + calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl,
        num_slices=num_slices, sum_fabe=false, sheets=true_partials)
    partial_be = [sum(pb) for pb=partial_be]
    
    # The lower limit of the contribution to the band energy from the partially occupied sheets
    # obtained using the regular occupations, the lower limit of the interval coefficients,
    # and the upper limit of the Fermi level interval
    if !flerrors
        partial_be₀ = fl*mesh_fa + calc_fabe(ebs, quantity="volume", ctype="min", fl=fl,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₁ = fl*mesh_fa + calc_fabe(ebs, quantity="volume", ctype="max", fl=fl,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₀ = [sum(pb) for pb=partial_be₀]; partial_be₁ = [sum(pb) for pb=partial_be₁]
    else
        partial_be₀ = fl₀*mesh_fa₀ + calc_fabe(ebs, quantity="volume", ctype="max", fl=fl₀,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₁ = fl₁*mesh_fa₁ + calc_fabe(ebs, quantity="volume", ctype="min", fl=fl₁,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₀ = [sum(pb) for pb=partial_be₀]; partial_be₁ = [sum(pb) for pb=partial_be₁]
    end

    # The contributions to the band energy error from the partially occupied quadratic triangles
    part_be_errs = (partial_be₁ .- partial_be₀)./2 # average error

    # The limits of the Fermi area interval
    ebs.fermiarea_interval = npg.*[fa₀,fa₁]
    # The Fermi area errors in each triangle
    ebs.fermiarea_errors = npg*[sum(m) for m=mesh_fa₋errs]
    ebs.fermilevel_interval = [fl₀,fl₁]
    ebs.fermilevel = fl
      
    # The limits of the band energy interval
    be₀ = 2*npg*(sum(sigma_be₀) + sum(partial_be₀))  
    be₁ = 2*npg*(sum(sigma_be₁) + sum(partial_be₁)) 
    ebs.bandenergy_interval = [be₀,be₁]
    ebs.bandenergy = 2*npg*(be + fl*epm.fermiarea/npg)

    ebs.partially_occupied = partial_occ
    ebs.sigma_bandenergy = 2*npg*sigma_be
    ebs.partial_bandenergy = 2*npg.*partial_be

    ebs.bandenergy_sigma_errors = 2*npg.*sigma_be_errs
    ebs.bandenergy_partial_errors = 2*npg.*part_be_errs
    ebs.bandenergy_errors = 2*npg.*(sigma_be_errs + part_be_errs)
    ebs
end

@doc """
    refine_mesh!(epm,ebs)

Perform one iteration of adaptive refinement. See the composite type
`bandstructure` for refinement options.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ebs::bandstructure`: a quadratic approximation of the band structure.

# Returns
- `ebs::bandstructure`: updated interval coefficients, Bezier coefficients, mesh, 
    extended mesh, Fermi level, band energy, ... for the quadratic approximation.

# Examples
```jldoctest
using Suppressor
using Pebsi.EPMs: m31
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!, refine_mesh!
epm = m31
ebs = init_bandstructure(epm);
@suppress calc_flbe!(epm,ebs)
@suppress refine_mesh!(epm,ebs)
abs(ebs.bandenergy - epm.bandenergy) < 1e-2
# output
true
```
"""
function refine_mesh!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure)
    spatial = pyimport("scipy.spatial")
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    err_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.target_accuracy
    faerr_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.fermiarea_eps
     
    n = def_min_split
    # Refine the tiles with the most error
    if ebs.refine_method == 1
        splitpos = sortperm(abs.(ebs.bandenergy_errors),rev=true)
        if length(splitpos) > n splitpos = splitpos[1:n] end

    # Refine the tiles with too much error (given the tiles' sizes).
    elseif ebs.refine_method == 2
        splitpos = filter(x -> x > 0,[abs(ebs.bandenergy_errors[i]) > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])

    # Refine a fraction of the number of tiles that have too much error.
    elseif ebs.refine_method == 3
        splitpos = filter(x -> x>0,[abs(ebs.bandenergy_errors[i]) > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])
        if length(splitpos) > n
            order = sortperm(abs.(ebs.bandenergy_errors[splitpos]),rev=true)
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        end 
    # Refine where the Fermi area errors are too much. Scale Fermi area errors
    # by Fermi level to get errors in terms of band energy
    elseif ebs.refine_method == 4
        bandenergy_err = 2*sum(ebs.fermiarea_errors)*ebs.fermilevel_interval[2]
        splitpos = filter(x -> x>0,[2*ebs.fermiarea_errors[i]*ebs.fermilevel_interval[2] > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])
        if length(splitpos) == 0 || bandenergy_err < ebs.target_accuracy
            ebs.refine_method = 3
            refine_mesh!(epm,ebs)
            return ebs
        elseif length(splitpos) > n
            order = sortperm(ebs.fermiarea_errors[splitpos],rev=true)
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        end
    # Refine any triangles with large Fermi area errors. There is no comparison 
    # against an allowed error. 
    elseif ebs.refine_method == 5
        if sum(ebs.fermiarea_errors) < ebs.fermiarea_eps
            println("Switching to band energy refinement.")
            ebs.refine_method = 7
            refine_mesh!(epm,ebs)
            return ebs
        end
        order = sortperm(abs.(ebs.fermiarea_errors),rev=true)
        if length(order) > n
            splitpos = order[1:round(Int,length(order)*def_frac_refined)]
        else
            splitpos = order
        end
    # Refine a fraction of triangles where the band energy errors are large. No 
    # comparison against an allowed error is performed.  
    elseif ebs.refine_method == 6
        # Only split triangles that are partially occupied
        splitpos = sort(unique([any(x->x==1,po) ? i : 0 for (i,po) in 
            enumerate(ebs.partially_occupied)]))[2:end]
        order = sortperm(abs.(ebs.bandenergy_errors[splitpos]),rev=true)
        if length(splitpos) > n
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        else
            splitpos = splitpos[order]
        end
    elseif ebs.refine_method == 7
        order = sortperm(abs.(ebs.bandenergy_errors),rev=true)
        if length(order) > n/def_frac_refined
            numsplit = round(Int,length(order)*def_frac_refined)
            splitpos = order[1:numsplit]
        elseif length(order) > n
            splitpos = order[1:n]
        else
            splitpos = order
        end        
    else
        ArgumentError("The refinement method has to be an integer 1,...,7.")
    end

    # Split fewer simplices if too many k-points are added.
    if ebs.stop_criterion == 4
        dim = size(epm.recip_latvecs,1)
        p = if dim == 2 3 else 6 end
        nkpts = size(ebs.eigenvals,2) - 2^dim
        max_added = p*length(splitpos)
        tot_kpts = nkpts + max_added
        if ebs.target_kpoints < tot_kpts
            numremove = round(Integer,(tot_kpts - ebs.target_kpoints)/p)
            if numremove >= length(splitpos) numremove = length(splitpos) - 1 end
            splitpos = splitpos[1:end-numremove]
        end
    end

    println("Number of split simplices: ", length(splitpos))
    if splitpos == []
        return ebs
    end

    dim = size(epm.recip_latvecs,1)
    centerpt = [1. / (dim+1) for i=1:dim+1]
    if dim == 2
        edgepts = [0 1/2 1/2; 1/2 0 1/2; 1/2 1/2 0]
    else
        edgepts = [1/2 1/2 1/2 0 0 0; 1/2 0 0 1/2 1/2 0; 0 1/2 0 1/2 0 1/2; 0 0 1/2 0 1/2 1/2]
    end
    
    # A single point at the center of the triangle
    if ebs.sample_method == 1
        new_meshpts = reduce(hcat,[barytocart(centerpt,s) for s=simplices[splitpos]])
    # Point at the midpoints of all edges of the triangle
    elseif ebs.sample_method == 2
        new_meshpts = reduce(hcat,[barytocart(edgepts,s) for s=simplices[splitpos]])
    # If the error is 2x greater than the tolerance, split edges. Otherwise,
    # sample at the center of the triangle.
    elseif ebs.sample_method == 3
        sample_type = [
            abs(ebs.bandenergy_errors[i]) > def_allowed_err_ratio*err_cutoff[i] ? 2 : 1 for i=splitpos]
        new_meshpts = reduce(hcat,[sample_type[i] == 1 ? 
        barytocart(centerpt,simplices[splitpos[i]]) :
        barytocart(edgepts,simplices[splitpos[i]])
        for i=1:length(splitpos)])
    else
        ArgumentError("The sample method for refinement has to be an integer with a value of 1,...,3.")
    end

    # Remove duplicates from the new mesh points.
    new_meshpts = unique_points(new_meshpts,rtol=ebs.rtol,atol=ebs.atol)
    new_eigvals = eval_epm(new_meshpts,epm,rtol=ebs.rtol,atol=ebs.atol)
    ebs.eigenvals = [ebs.eigenvals new_eigvals]

    println("Unique points added: $(size(new_meshpts,2))")

    # There should technically be an additional step at this point where
    # symmetrically equivalent points are removed from `new_meshpts` (points
    # on different boundaries of the IBZ may be rotationally or translationally
    # equivalent, but I figure the chances of two points being equivalent are
    # pretty slim and the extra cost isn't too great, but I could be wrong, so 
    # I'm making a note.

    cv_pointsᵢ = get_cvpts(ebs.mesh,epm.ibz,atol=ebs.atol)
    # Calculate the maximum distance between neighboring points
    bound_limit = def_max_neighbor_tol*maximum(
        reduce(vcat,[[norm(ebs.mesh.points[i,:] - ebs.mesh.points[j,:]) 
                    for j=get_neighbors(i,ebs.mesh,ebs.num_near_neigh)] for i=cv_pointsᵢ]))

    if dim == 2
        # The Line segments that bound the IBZ.
        borders = [Matrix(epm.ibz.points[i,:]') for i=epm.ibz.simplices]
        distfun = lineseg₋pt_dist
        # Translations that need to be considered when calculating points outside the IBZ.
        # Assumes the reciprocal latice vectors are Minkowski reduced.
        bztrans = [[[i,j] for i=-1:1,j=-1:1]...]
    else
        borders = [Matrix(epm.ibz.points[f,:]') for f=get_uniquefacets(epm.ibz)]
        distfun = ptface_mindist
        bztrans = [[[i,j,k] for i=-1:1,j=-1:1,k=-1:1]...]
    end
     
    # The number of points in the mesh before adding new points.
    s = size(ebs.mesh.points,1)
    m = maximum(ebs.sym₋unique)

    # Indices of the new mesh points.
    new_ind = (m+1):(m+size(new_meshpts,2))

    # Indices of sym. equiv. points on and nearby the boundary of the IBZ. Pointer to the symmetrically unique points.
    sym_ind = zeros(Int,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))
     
    # Keep track of points on the IBZ boundaries.
    nₘ = 0
    # Add points to the mesh on the boundary of the IBZ.
    neighbors = zeros(Float64,dim,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))

    for i=1:length(new_ind),op=epm.pointgroup,trans=bztrans
        pt = op*new_meshpts[:,i] + epm.recip_latvecs*trans
        if (any([isapprox(distfun(pt,border),0,atol=ebs.atol) for border=borders]) && 
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                        [ebs.mesh.points' new_meshpts neighbors[:,1:nₘ]],dims=1)))
            nₘ += 1
            sym_ind[nₘ] = new_ind[i]
            neighbors[:,nₘ] = pt
        end
    end
    @show nₘ
    if m == s
        ebs.mesh = spatial.Delaunay([ebs.mesh.points; new_meshpts'; neighbors[:,1:nₘ]'])
    else
        ebs.mesh = spatial.Delaunay([ebs.mesh.points[1:m,:]; new_meshpts'; neighbors[:,1:nₘ]';
            ebs.mesh.points[m+1:end,:]])        
    end

    # Add points to the extended mesh nearby but outside of the IBZ
    nₑ = nₘ
    for i=1:length(new_ind),op=epm.pointgroup,trans=bztrans
        pt = op*new_meshpts[:,i] + epm.recip_latvecs*trans
        if (any([distfun(pt,border) < bound_limit for border=borders]) &&
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                    [ebs.ext_mesh.points' new_meshpts neighbors[:,1:nₑ]],dims=1)))
            nₑ += 1
            sym_ind[nₑ] = new_ind[i]
            neighbors[:,nₑ] = pt
        end
    end

    @show nₑ

    if m == s
        ebs.sym₋unique = [ebs.sym₋unique[1:m]; new_ind; sym_ind[1:nₑ]] 
        ebs.ext_mesh = spatial.Delaunay([ebs.ext_mesh.points[1:m,:]; new_meshpts'; neighbors[:,1:nₑ]'])
    else
        ebs.sym₋unique = [ebs.sym₋unique[1:m]; new_ind; sym_ind[1:nₘ]; ebs.sym₋unique[m+1:end];
            sym_ind[nₘ+1:nₑ]]
        ebs.ext_mesh = spatial.Delaunay([ebs.ext_mesh.points[1:m,:]; new_meshpts'; neighbors[:,1:nₘ]';
            ebs.ext_mesh.points[m+1:end,:]; neighbors[:,nₘ+1:nₑ]']) 
    end

    ebs.simplicesᵢ = notbox_simplices(ebs.mesh)
    coeffs = [get_intercoeffs(index,ebs.mesh,ebs.ext_mesh,
    ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,
    neighbor_method=ebs.neighbor_method,epm=epm) for index=1:length(ebs.simplicesᵢ)]

    ebs.mesh_intcoeffs = [coeffs[index][1] for index=1:length(ebs.simplicesᵢ)]
    ebs.mesh_bezcoeffs = [coeffs[index][2] for index=1:length(ebs.simplicesᵢ)]        
    ebs
end

@doc """
    get_tolerances(epm,ebs)

Calculate the Fermi level and Fermi area tolerances.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ebs::bandstructure`: a quadratic approximation of the band structure.

# Returns
- `db::Real`: the derivative of the band energy with respect to the Fermi level.
- `da::Real`: the derivative of the Fermi area with respect to the Fermi level.
- `dltol::Real`: the Fermi level tolerance.
- `datol::Real`: the Fermi area tolerance.

# Examples
```jldoctest
using Pebsi.EPMs: m41
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!, get_tolerances
epm = m41
ebs = init_bandstructure(epm);
calc_flbe!(epm,ebs)
tol = get_tolerances(epm,ebs)
length(tol)
# output
4
```
"""
function get_tolerances(epm,ebs; num_slices=def_num_slices)
    dim = size(epm.recip_latvecs,1)
    start = 2^dim + 1
    stepsize = (maximum(ebs.eigenvals[:,start:end]) - 
       minimum(ebs.eigenvals[:,start:end]))*def_deriv_step
    numsteps = 4 # number of band energies/Fermi areas needed to compute derivatives
    es = collect(-numsteps/2*stepsize:stepsize:numsteps/2*stepsize) .+ ebs.fermilevel
    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(dim,2) 
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]

    npg = length(epm.pointgroup); bens = []; fas = []
    for fl=es
        be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        fa = calc_fabe(ebs, quantity="area", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        # be = 2*npg*(be + fl*epm.fermiarea/npg)
        # Removed translation of band energy because numbers close to zero throw off the derivative
        be = 2*npg*be
        push!(bens,be); push!(fas,fa)
    end
    dbs = [(-bens[i+2] + 8*bens[i+1] - 8*bens[i-1] + bens[i-2])/(es[i+2]-es[i-2]) for i=3:length(bens)-2]
    das = [(-fas[i+2] + 8*fas[i+1] - 8*fas[i-1] + fas[i-2])/(es[i+2]-es[i-2]) for i=3:length(fas)-2];
    db = maximum(abs.(das)); da = maximum(abs.(dbs))
    dltol = ebs.target_accuracy/db; datol = da*dltol
    db,da,dltol,datol
end

@doc """
    stop_refinement(ebs)

Select a condition that determines if refinement may stop.

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: a empirical pseudopotential model.
- `ebs::bandstructure`: a band structure object

# Returns
- `stop::Bool`: a boolean that tells when refinement may stop.

# Examples
```
using Pebsi.QuadraticIntegration, Pebsi.EPMs
epm = m11
ebs = init_bandstructure(epm)
calc_flbe!(epm,ebs)
stop_refinement!(ebs)
```
"""
function stop_refinement!(epm::Union{epm₋model,epm₋model2D},ebs::bandstructure,
    prevbe)::Bool
    stop = false
    if ebs.stop_criterion == 1
        stop = abs(sum(ebs.bandenergy_errors)) < ebs.target_accuracy
    elseif ebs.stop_criterion == 2
        stop = abs(ebs.bandenergy - prevbe) < ebs.target_accuracy
    elseif ebs.stop_criterion == 3
        db,da,dltol,datol = get_tolerances(epm,ebs)
        ebs.fermiarea_eps = datol
        stop = db/da*diff(ebs.fermiarea_interval)[1]/2 < ebs.target_accuracy
    elseif ebs.stop_criterion == 4 
        nkpts = size(ebs.eigenvals,2) - 2^size(epm.recip_latvecs,1)
        stop = ((ebs.target_kpoints - nkpts) < def_kpoint_tol*nkpts/100) || (nkpts >  ebs.target_kpoints)
    else
        error("Valid values of stop_criterion are integers 1,...,4")
    end
    stop
end

@doc """
    quadratic_method(epm,stop_criterion,init_msize,num_near_neigh,num_neighbors,
        fermiarea_eps,target_accuracy,fermilevel_method,refine_method,sample_method,
        neighbor_method,fatten,rtol,atol,uniform)

Calculate the band energy using uniform or adaptive quadratic integation.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential
- `stop_criterion::Integer=3`: sets the condition that stops adaptive refinement 

See the documentation for `bandstructure` for descriptions of the optional arguements.

# Returns
- `ebs::bandstructure`: a quadratic approximation of the band structure.

# Examples
```jldoctest
using Pebsi.EPMs: m51
using Pebsi.QuadraticIntegration: quadratic_method
epm = m51
ebs = quadratic_method(epm,target_accuracy=1e-2)
abs(ebs.bandenergy - epm.bandenergy) < 1e-1
# output
true
```
"""
function quadratic_method(epm::Union{epm₋model2D,epm₋model};
    init_msize::Int=def_init_msize, num_near_neigh::Int=def_num_near_neigh,
    num_neighbors::Union{Nothing,Int}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Integer=def_fermilevel_method, 
    refine_method::Integer=def_refine_method,
    sample_method::Integer=def_sample_method, 
    neighbor_method::Integer=def_neighbor_method,
    fatten::Real=def_fatten, rtol::Real=def_rtol, atol::Real=def_atol,
    uniform::Bool=def_uniform, weighted::Bool=def_weighted,
    constrained::Bool=def_constrained, stop_criterion::Integer=def_stop_criterion,
    target_kpoints::Integer=def_target_kpoints)::bandstructure

    dim = size(epm.recip_latvecs,1)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end

    ebs = init_bandstructure(epm,init_msize=init_msize, num_near_neigh=num_near_neigh,
        num_neighbors=num_neighbors,fermiarea_eps=fermiarea_eps, 
        target_accuracy=target_accuracy, fermilevel_method=fermilevel_method, 
        refine_method=refine_method, sample_method=sample_method, 
        neighbor_method=neighbor_method, fatten=fatten, weighted=weighted, 
        constrained=constrained, stop_criterion=stop_criterion,
        target_kpoints=target_kpoints, rtol=rtol, atol=atol)
    calc_flbe!(epm,ebs)
    if uniform return ebs end
    counter = 0; prevbe = 1e9; tmp = []
    sd = 3 # rounding parameter for print statements
    stop = stop_refinement!(epm,ebs,prevbe)
    while !stop
        counter += 1; refine_mesh!(epm,ebs); calc_flbe!(epm,ebs)
        stop = stop_refinement!(epm,ebs,prevbe)
        prevbe = ebs.bandenergy 
        if counter > max_refine_steps
            @warn "Failed to calculate the band energy to within the desired accuracy $(ebs.target_accuracy) after $(max_refine_steps) iterations."
            break
        end
        diffbe = abs(ebs.bandenergy - prevbe)
        ϵᵦ = abs(ebs.bandenergy - epm.bandenergy) 
        ϵₗ = abs(ebs.fermilevel - epm.fermilevel)
        db,da,dltol,datol = get_tolerances(epm,ebs)
        println("Number of simplices: ", length(ebs.simplicesᵢ)) 
        # println("True errors")
        # println("ϵᵦ: ", round(ϵᵦ,sigdigits=sd))
        # println("ϵₗ: ", round(ϵₗ,sigdigits=sd))

        # println("Estimated band energy error")
        # println("δB/ϵᵦ: ", round(diffbe/ϵᵦ,sigdigits=sd))
        # println("ΔB/ϵᵦ: ", round(sum(ebs.bandenergy_errors)/ϵᵦ,sigdigits=sd))
        # println("dB/dL*ΔLₜ/ϵᵦ: ", round(db*diff(ebs.fermilevel_interval)[1]/2/ϵᵦ,sigdigits=sd))
        # println("dB/dA*ΔAₜ/ϵᵦ: ", round(db/da*diff(ebs.fermiarea_interval)[1]/2/ϵᵦ,sigdigits=sd))
        
        # println("Estimated Fermi level error")
        # println("ΔL/ϵₗ: ", round(diff(ebs.fermilevel_interval)[1]/2/ϵₗ,sigdigits=sd))
        # println("Estimated Fermi area errors")
        # println("ΔA: ", round(diff(ebs.fermiarea_interval)[1]/2,sigdigits=sd))
        # println("dA/dL*ΔLₜ: ", round(da*diff(ebs.fermilevel_interval)[1]/2,sigdigits=sd))
        
        # println("Derivatives and tolerances")
        # println("dB/dL dA/dL dB/dA ΔAₜ ΔLₜ = ",round.([db,da,db/da,dltol,datol],sigdigits=sd),"\n")
    end
    ebs
end

"""
    truebe(epm,ebs,ndivs,num_cores,triangles)
    
Calculate (roughly) the true band energy error for each quadratic triangle.

# Arguments
- `epm::bandstructure`: a quadratic approximation of the band structure
- `ebs::epm₋model2D`: an empirical pseudopotential
- `ndivs::Integer`: the number of divisions of triangles when computing the band
    energy component within each triangle using the rectangular method with a triangular
    base.
- `num_cores::Integer=1`: the number of cores to use when computing in parallel.
- `triangles`: a list of triangles over which to compute the "true" band energy. If
    nothing is provided, compute over all triangles in the approximation.

# Returns
- `sigma_be::Vector{Real}`: the true sigma band energy over each triangle.
- `part_be::Vector{Real}`: the true partial band energy over each triangle.

# Examples
```
using Pebsi.EPMs: m51
using Pebsi.QuadraticIntegration: quadratic_method, init_bandstructure, calc_flbe!, truebe
epm = m51
ebs = init_bandstructure(epm)
ebs = calc_flbe!(epm,ebs)
sigma_be,part_be = truebe(epm,ebs,10)
```
"""
function truebe(epm::epm₋model2D,ebs::bandstructure,ndivs::Integer;
    num_cores::Integer=1,triangles::Union{Nothing,Integer}=nothing)
    dim = 2
    deg = 2
    num_tri = length(ebs.simplicesᵢ)
    num_sheets = epm.sheets
    
    if triangles == nothing
        triangles = 1:num_tri
    end

    # Locate the highest occupied sheet
    max_sheet = 0
    for tri = triangles
        for sheet=1:num_sheets
            if ebs.partially_occupied[tri][sheet] == 1
                if sheet > max_sheet max_sheet = sheet end
            end
        end
    end
    max_sheet += 2

    # Identify occupied, partially occupied, and unoccupied sheets
    bpts = trimesh(ndivs)
    vals = zeros(max_sheet,size(bpts,2))
    pts = zeros(3,length(ebs.simplicesᵢ))
    sigma_be = [0. for _=1:length(triangles)]
    part_be = [0. for _=1:length(triangles)]
    da,last,ps = 0,0,[]
    # partocc = [[0 for _=1:max_sheet] for _=1:num_tri]
    partocc = [0 for _=1:max_sheet]
    for (i,tri)=enumerate(triangles)
        pts = barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[tri],:]')

        if num_cores == 1
            vals = eval_epm(pts,epm,sheets=max_sheet)
        else
            vals = reduce(hcat,pmap(x->eval_epm(x,epm,sheets=max_sheet),
                [pts[:,i] for i=1:size(pts,2)]))
        end

        for sheet = 1:max_sheet
            if all((vals[sheet,:] .< ebs.fermilevel) .| isapprox.(vals[sheet,:],epm.fermilevel))
                partocc[sheet] = 0
            elseif all((vals[sheet,:] .> ebs.fermilevel) .| isapprox.(vals[sheet,:],epm.fermilevel))
                partocc[sheet] = 2
            else
                partocc[sheet] = 1
            end
        end
         
        da = simplex_size(ebs.mesh.points[ebs.simplicesᵢ[tri],:]')/ndivs^2
        last = findlast(x->x==0,partocc)
        if last != nothing 
            sigma_be[i] = 2*length(epm.pointgroup)*sum(vals[1:last,:])*da
        end
        ps = findall(x->x==1,partocc)
        if length(ps) != 0
            part_be[i] = 2*length(epm.pointgroup)*sum(filter(x->(x<ebs.fermilevel || isapprox(x,ebs.fermilevel)), 
                    vals[ps,:]))*da
        end
    end
      
    approx_be = sum(sigma_be) + sum(part_be)
    approx_err = round(abs(approx_be - epm.bandenergy),sigdigits=3)
    println("The band energy error with triangular integration is $(approx_err)")
    sigma_be,part_be
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
2-element Vector{Real}:
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
 1
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
function analytic_area1D(coeffs::AbstractVector{<:Real},limits::AbstractVector)::Real
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
    simpson3D(coeffs,tetrahedron,quantity;num_slices,values,gauss,split,corner,atol,rtol)

Calculate the volume or hypervolume beneath a quadratic within a tetrahedron.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial 
    over the tetrahedron.
- `tetrahedron::AbstractMatrix{<:Real}`: the Cartesian coordinates of the point at the corner of the tetrahedron.
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
tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
coeffs = [-1/10, -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
simpson3D(coeffs,tet,"area",num_slices=10) ≈ 0.01528831567698499
# output
true
```
"""
function simpson3D(coeffs::AbstractVector{<:Real}, tetrahedron::AbstractMatrix{<:Real},
    quantity::String; num_slices::Integer=def_num_slices, values::Bool=false, gauss::Bool=true,
    split::Bool=true, corner::Union{Nothing,Integer}=nothing,atol::Real=def_atol)
    dim = 3; deg = 2
     
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
            intvals[i] = quad_area₋volume([pts; coeffs2D'],quantity)
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
    containment_percentage(epm,ebs,divs,atol)

Calculate the containment percentage of a quadratic interval representation of the bandstructure.

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: an empirical pseudopotential model.
- `ebs::bandstructure`: a `bandstructure` data structure.
- `divs::Integer`: the number of divisions of a triangle when sampling quadratic surfaces
- `atol::Real=1e-6`: an absolute tolerance. Eigenvalues closer than `atol` to the
    interval are considered contained.

# Returns
- `percent::Real`: the percentage of eigenvalues contained by interval quadratics.
- `relerror::Vector{<:Real}`: the minimum distance of a eigenvalue to the interval
    quadratic divided by the size of the interval.

# Examples
```
using Pebsi.QuadraticIntegration, Pebsi.EPMs
epm = Al_epm
ebs = init_bandstructure(epm)
divs = 3
containment_percentage(epm,ebs,divs)
```
"""
function containment_percentage(epm::Union{epm₋model,epm₋model2D},
    ebs::bandstructure,divs::Integer,atol::Real=1e-6)
    dim = size(epm.recip_latvecs,1); deg = 2
    simplex_bpts = sample_simplex(dim,deg)
    bpts = sample_simplex(dim,divs)
    npts = size(bpts,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    relerror = zeros(2*length(ebs.simplicesᵢ)*size(bpts,2)*epm.sheets)
    rdiff₁ = 0; rdiff₂ = 0; interval = [0.,0.]; ind = 0
    vals₁ = zeros(npts); vals₂ = zeros(npts)
    for tri=1:length(ebs.simplicesᵢ)
        pts = barytocart(bpts,simplices[tri])
        evals = eval_epm(pts,epm,rtol=ebs.rtol,atol=ebs.atol)
        for sheet=1:epm.sheets
            interval = ebs.mesh_intcoeffs[tri][sheet]
            vals₁ = eval_poly(bpts,interval[1,:],dim,deg)
            vals₂ = eval_poly(bpts,interval[2,:],dim,deg)
            for i=1:npts
                ind = (2*i-1) + 2*(sheet-1)*npts + 2*(tri-1)*(epm.sheets*npts)
                rdiff₁ = abs(evals[sheet,i] - vals₁[i])/(vals₂[i] - vals₁[i])
                if rdiff₁ < 0 && !isapprox(rdiff₁,0,atol=atol)
                    relerror[ind] = -rdiff₁
                end
                rdiff₂ = (vals₂[i] - evals[sheet,i])/(vals₂[i] - vals₁[i])
                if rdiff₂ < 0 && !isapprox(rdiff₂,0,atol=atol)
                    relerror[ind+1] = -rdiff₂
                end
            end
        end
    end
    filter!(x -> x!=0, relerror)
    percent = (1-length(relerror)/(length(ebs.simplicesᵢ)*size(bpts,2)*epm.sheets))*100
    percent,relerror
end

do_nothing(x;dims=0) = x

@doc """
    calc_fabe(ebs,quantity,ctype,fl;num_slices,sum_fabe,sheets)

Calculate the Fermi area or band energy for a candidate Fermi level per patch or the sum.

# Arguments
- `ebs::bandstructure`: a `bandstructure` composite type.
- `quantity::String`: the quantity calculated ("area" or "volume")
- `ctype::String`: determines which coefficients to use. Options include "min",
    "max", and "mean". The least-squares coefficients are used with option "mean".
- `fl::Real=ebs.fermilevel`: the candidate Fermi level.
- `num_slices::Integer=def_num_slices`: the number of slices for integration in 3D.
- `sum_fabe::Bool=true`: if true, sum the Fermi area or band energy of all patches.
- `sheets::Vector:` the sheets per triangle to consided as a vector of vectors.

# Returns
- `fabe::Real`: the Fermi level or band energy

# Examples
```
using Pebsi.EPMs, Pebsi.QuadraticIntegration
epm = m11
ebs = init_bandstructure(epm)
quantity = "area"
fl = epm.fermilevel
ctype = "mean"
calc_fabe(ebs,quantity,ctype,fl)
```
"""
function calc_fabe(ebs::bandstructure; quantity::String, ctype::String, fl::Real=ebs.fermilevel,
    num_slices::Integer=def_num_slices, sum_fabe::Bool=true, sheets::Union{Nothing,Vector}=nothing)

    ns = length(ebs.simplicesᵢ)
    nsheets = size(ebs.eigenvals,1)
    if sheets == nothing
        sheets = [collect(1:nsheets) for i=1:ns]
    end
    dim = length(ebs.simplicesᵢ[1]) - 1
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    if ctype == "min" || ctype == "max"
        coeffs = ebs.mesh_intcoeffs
        cfun = if ctype == "min" minimum else maximum end
    else
        coeffs = ebs.mesh_bezcoeffs
        cfun = do_nothing
    end

    fabe = [[0. for sheet=1:nsheets] for tri=1:ns]
    if dim == 2
        simplex_bpts = sample_simplex(dim,2)
        simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
        for tri=1:ns
            for sheet=sheets[tri]
                fabe[tri][sheet] = quad_area₋volume([simplex_pts[tri]; 
                    [cfun(coeffs[tri][sheet],dims=1)...]' .- fl],quantity)
            end
        end
    else
        for tri=1:ns
            for sheet=sheets[tri]
                fabe[tri][sheet] = simpson3D(vec(cfun(coeffs[tri][sheet] .- fl, dims=1)),
                    simplices[tri], quantity, num_slices=num_slices)
            end
        end
    end
    if sum_fabe sum(sum(fabe)) else fabe end
end
end # module