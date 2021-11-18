module QuadraticIntegration

using SymmetryReduceBZ.Utilities: unique_points, shoelace, remove_duplicates, get_uniquefacets
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ..Polynomials: eval_poly,getpoly_coeffs,getbez_pts₋wts,eval_bezcurve,
    conicsection, eval_1Dquad_basis, evalpoly1D, get_1Dquad_coeffs,
    solve_quadratic
using ..EPMs: eval_epm, RytoeV, epm₋model, epm₋model2D
using ..Mesh: get_neighbors,notbox_simplices,get_cvpts,ibz_init₋mesh, 
    get_extmesh, choose_neighbors, choose_neighbors3D, trimesh
using ..Geometry: order_vertices!,simplex_size,insimplex,barytocart,carttobary,
    sample_simplex,lineseg₋pt_dist, mapto_xyplane, ptface_mindist
using ..Defaults

using QHull: chull,Chull
using LinearAlgebra: cross,det,norm,dot,I,diagm,pinv
using Statistics: mean
using Base.Iterators: flatten
using SparseArrays: findnz
using PyCall: PyObject, pyimport
using Distributed: pmap
using FastGaussQuadrature: gausslegendre

export bandstructure, init_bandstructure, quadval_vertex, corner_indices, 
    edge_indices, simplex_intersects, saddlepoint, split_bezsurf₁, 
    split_bezsurf, analytic_area, analytic_volume, sub₋coeffs,
    two₋intersects_area₋volume, quad_area₋volume, get_intercoeffs, calc_fl,
    calc_flbe!, refine_mesh!, get_tolerances, quadratic_method!, truebe, 
    bezcurve_intersects, getdomain, analytic_area1D, simpson, simpson2D, 
    linept_dist, tetface_areas, simpson3D, quadslice_tanpt

@doc """
    bandstructure

A container for all variables related to the band structure.

# Arguments
- `init_msize::Int`: the initial size of the mesh over the IBZ. The number of 
    points is approximately proportional to init_msize^2/2.
- `num_near_neigh::Int`: the number of nearest neighbors to consider to include when
    calculating interval coefficients.
- `fermiarea_eps::Real`: a tolerance used during the bisection algorithm that 
    determines how close the midpoints of the Fermi area interval is to the true
    Fermi area.
- `target_accuracy::Real`: the accuracy desired for the band energy at the end 
    of the computation.
- `fermilevel_method::Int`: the method for computing the Fermi level.
    1- bisection, 2-Chandrupatla.
- `refine_method::Int`: the method of refinement. 1-refine the tile with the most
    error. 2-refine the tiles with too much error given the sizes of the tiles.
- `sample_method::Int`: the method of sampling a tile with too much error. 1-add
    a single point at the center of the triangle. 2-add points the midpoints of 
    all three edges.
- `neighbor_method::Int`: the method for selecting neighboring points in the 
    calculion of interval coefficients.
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.
- `mesh::PyObject`: a Delaunay triangulation of points over the IBZ.
- `simplicesᵢ::Vector{Vector{Int}}`: the indices of points at the corners of the 
    tile for all tiles in the triangulation.
- `ext_mesh::PyObject`: a Delaunay triangulation of points within and around the 
    IBZ. The number of points outside is determined by `num_near_neigh`.
- `sym₋unique::AbstractVector{<:Int}`: the indices of symmetrically unique points
    in the mesh.
- `eigenvals::AbstractMatrix{<:Real}`: the eigenvalues at each of the points unique
    by symmetry.
- `fatten`:: a variable that scales the size of the interval coefficients.
- `mesh_intcoeffs::Vector{Vector{Matrix{Float64}}}`:the interval Bezier 
    coefficients for all tiles and sheets.
- `mesh_bezcoeffs::Vector{Vector{Vector{Float64}}}`: the least-squares Bezier
    coefficients for all tiles and sheets.
- `fermiarea_interval::AbstractVector{<:Real}`: the Fermi area interval. 
- `fermilevel_interval::AbstractVector{<:Real}`: the Fermi level interval. 
- `bandenergy_interval::AbstractVector{<:Real}`: the band energy interval.
- `fermilevel::Real`: the true Fermi level.
- `bandenergy::Real`: the true band energy.
- `partially_occupied::Vector{Vector{Int64}}`: the sheets that are partially
    occupied in each tile.
- `bandenergy_errors::Vector{<:Real}`: the band energy errors for each tile
    in the triangulation.
- `fermiarea_errors::Vector{<:Real}`: the Fermi area errors for each tile in the
    triangulation.

"""
mutable struct bandstructure
    init_msize::Int
    num_near_neigh::Int
    num_neighbors::Int
    fermiarea_eps::Real
    target_accuracy::Real
    fermilevel_method::Int
    refine_method::Int
    sample_method::Int
    neighbor_method::Int
    rtol::Real
    atol::Real
    mesh::PyObject
    simplicesᵢ::Vector{Vector{Int}}
    ext_mesh::PyObject
    sym₋unique::AbstractVector{<:Int}
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
end

@doc """
    init_bandstructure(epm; init_msize,num_near_neigh,fermiarea_eps,target_accuracy,
        fermilevel_method,refine_method,sample_method,fatten,rtol,atol)

Initialize a band structure container.

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: an empirical pseudopotential. 
- `init_msize::Int`: the initial size of the mesh over the IBZ. The number of 
    points is approximately proportional to init_msize^2/2.
- `num_near_neigh::Int`: the number of nearest neighbors to consider to include when
    calculating interval coefficients.
- `fermiarea_eps::Real`: a tolerance used during the bisection algorithm that 
    determines how close the midpoints of the Fermi area interval is to the true
    Fermi area.
- `target_accuracy::Real`: the accuracy desired for the band energy at the end 
    of the computation.
- `fermilevel_method::Int`: the method for computing the Fermi level.
    1- bisection, 2-Chandrupatla.
- `refine_method::Int`: the method of refinement. 1-refine the tile with the most
    error. 2-refine the tiles with too much error given the sizes of the tiles.
    3-refine the tiles with too much error given the sizes of the tiles and 
    split tiles with less error only once (add a sample point at the center of 
    the tile instead at each edge midpoint).
- `sample_method::Int`: the method of sampling a tile with too much error. 1-add
    a single point at the center of the triangle. 2-add point the midpoints of 
    all three edges.
- `neighbor_method::Int`: the method of selecting neighboring points for the 
    calculation of interval coefficients.
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.

# Returns
- `::bandstructure`: a container containing useful information about the 
    band structure representation.

# Examples
```
import Pebsi.EPMs: m11
import Pebsi.QuadraticIntegration: init_bandstructure
init_bandstructure(m11)
```
"""
function init_bandstructure(
    epm::Union{epm₋model,epm₋model2D,epm₋model};
    init_msize::Int=def_init_msize,
    num_near_neigh::Int=def_num_near_neigh,
    num_neighbors::Union{Nothing,Int}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Int=def_fermilevel_method,
    refine_method::Int=def_refine_method,
    sample_method::Int=def_sample_method,
    neighbor_method::Int=def_neighbor_method,
    fatten::Real=def_fatten,
    rtol::Real=def_rtol,
    atol::Real=def_atol)

    dim = size(epm.recip_latvecs,1)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end

    mesh = ibz_init₋mesh(epm.ibz,init_msize;rtol=rtol,atol=atol)
    mesh,ext_mesh,sym₋unique = get_extmesh(epm.ibz,mesh,epm.pointgroup,
        epm.recip_latvecs,num_near_neigh; rtol=rtol,atol=atol)
    simplicesᵢ = notbox_simplices(mesh)
 
    uniqueᵢ = sort(unique(sym₋unique))[2:end]
    # eigenvals = zeros(Float64,epm.sheets,size(mesh.points,1))
    estart = if dim == 2 4 else 8 end
    eigenvals = zeros(Float64,epm.sheets,estart+length(uniqueᵢ))
    for i=uniqueᵢ
        eigenvals[:,i] = eval_epm(mesh.points[i,:], epm, rtol=rtol, atol=atol)
    end

    coeffs = [get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,
        simplicesᵢ,fatten,num_near_neigh,epm=epm,neighbor_method=neighbor_method) for index=1:length(simplicesᵢ)]

    mesh_intcoeffs = [coeffs[index][1] for index=1:length(simplicesᵢ)]
    mesh_bezcoeffs = [coeffs[index][2] for index=1:length(simplicesᵢ)]
    
    partially_occupied = [zeros(Int,epm.sheets) for _=1:length(simplicesᵢ)]
    bandenergy_errors = zeros(length(simplicesᵢ))
    bandenergy_sigma_errors = zeros(length(simplicesᵢ))
    bandenergy_partial_errors = zeros(length(simplicesᵢ))
    fermiarea_errors = zeros(length(simplicesᵢ))
     
    sigma_bandenergy = zeros(length(simplicesᵢ))
    partial_bandenergy = zeros(length(simplicesᵢ))
     
    fermiarea_interval=[0,0]
    fermilevel_interval=[0,0]
    bandenergy_interval=[0,0]
    fermilevel=0
    bandenergy=0
        
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
        fermiarea_errors)
end   

# @doc """
#     simpson(interval_len,vals)

# Calculate the integral of a univariate function with the composite Simpson's method

# # Arguments
# - `interval_len::Real`: the length of the inteval the functios is integrated over.
# - `vals::AbstractVector{<:Real}`: the value of the function on a uniform, closed 
#     grid over the interval.

# # Returns
# - `::Real`: the approximate integral of the function over the iterval.

# # Examples
# ```jldoctest
# import Pebsi.QuadraticIntegration: simpson
# num_intervals = 20
# f(x) = x^5 - x^4 - 2*x^3
# vals = map(x->f(x),collect(0:1/(2*num_intervals):1))
# interval_len = 1
# answer = (-8/15)
# abs(simpson(interval_len,vals) - answer)
# # output
# 7.812500002479794e-8
# ```
# """
# function simpson(interval_len::Real,vals::AbstractVector{<:Real})::Real
#     num_intervals = Int((length(vals) - 1)/2)
#     simp_wts = ones(Int,2*num_intervals+1)
#     simp_wts[2:2:end-1] .= 4
#     simp_wts[3:2:end-2] .= 2 
#     interval_len/(6*num_intervals)*dot(simp_wts,vals)
# end

@doc """
    quadval_vertex(bezcoeffs)

Calculate the value of a quadratic curve at its vertex.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the quadratic polynomial coefficients.
"""
function quadval_vertex(bezcoeffs::AbstractVector{<:Real})::Real
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
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
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `intersects::Array`: the intersections organized by edge in a 1D array. Each 
    element of the array is a 2D array where the columns are the Cartesian
    coordinates of intersections.

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
        # @show edge_bezpts
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
    saddlepoint(coeffs)

Calculate the saddle point of a quadratic Bezier surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial.
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `::AbstractVector{<:Real}`: the coordinates of the saddle point in Barycentric coordinates.

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
    split_bezsurf₁(bezpts,atol)

Split a Bezier surface once into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

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
function split_bezsurf₁(bezpts::AbstractMatrix{<:Real},
    allpts::AbstractArray=[]; atol::Real=def_atol)::AbstractArray
    spatial = pyimport("scipy.spatial")
    dim = 2
    deg = 2
    triangle = bezpts[1:end-1,corner_indices]
    if simplex_size(triangle) < def_min_simplex_size
        return [bezpts]
    end

    coeffs = bezpts[end,:]
    pts = bezpts[1:end-1,:]
    simplex_bpts = sample_simplex(dim,deg)
    intersects = simplex_intersects(bezpts,atol=atol)
    spt = saddlepoint(coeffs)
    if intersects == [[],[],[]]
        if insimplex(spt)
            allpts = [pts barytocart(spt,triangle)]
        else
            allpts = pts
        end
    else
        allintersects = reduce(hcat,[i for i=intersects if i!=[]])
        if insimplex(spt) # Using the default absolute tolerance 1e-12
            allpts = [pts barytocart(spt,triangle) allintersects]
        else
            allpts = [pts allintersects]
        end
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
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_bezsurf
bezpts = [0. 0.5 1 0.5 1 1; 0. 0. 0. 0.5 0.5 1; 1.1 1.2 -1.3 1.4 1.5 1.6]
split_bezsurf(bezpts)
# output
1-element Vector{Matrix{Float64}}:
 [0.0 0.5 … 1.0 1.0; 0.0 0.0 … 0.5 1.0; 1.1 1.2 … 1.5 1.6]
```
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

Calculate the area within a triangle and a canonical, rational, Bezier curve.

# Arguments
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic, Bezier curve.
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
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratica surface.
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic, Bezier curve.
- `atol::Real=1e-9`: an absolute tolerance for finite precision tolerances.

# Returns
- `::Real`: the area within the triangle and Bezier curve.

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
    two₋intersects_area₋volume(bezpts,quantity;atol=1e-9)

Calculate the area or volume within a quadratic curve and triangle and Quadratic surface.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of a quadratic surface.
- `quantity::String`: the quantity to compute ("area" or "volume").
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
    quantity::String; atol::Real=def_atol)::Real
     
    # Calculate the bezier curve and weights, make sure the curve passes through
    # the triangle
    triangle = bezpts[1:end-1,corner_indices]
    coeffs = bezpts[end,:]
    intersects = simplex_intersects(bezpts,atol=atol)

    # No intersections
    if intersects == [[],[],[]]
        # Case where the sheet is completely above or below 0.    
        if all(bezpts[end,:] .< 0) && !all(isapprox.(bezpts[end,:],0,atol=atol))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
            return areaₒᵣvolume
        end
        if all(bezpts[end,:] .> 0) && !all(isapprox.(bezpts[end,:],0,atol=atol))
            areaₒᵣvolume = 0
            return areaₒᵣvolume
        end
    end

    bezptsᵣ = []
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
        if !insimplex(carttobary(ptᵣ,triangle))
            bezwtsᵣ[2] *= -1
            ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
            if !insimplex(carttobary(ptᵣ,triangle))
                intersects = [[],[],[]]
            end
        end
    end

    # If the tangent lines are close to parallel, the middle Bezier point of the
    # curve will be very far away, which introduces numerical errors. We handle
    # this by splitting the surface up and recalculating.
    # Also, split the surface if the level curve isn't linear and the saddle point 
    # is within the triangle.
    cstype = conicsection(bezpts[end,:]) # using the default tolerance of 1e-12
    linear = any(cstype .== ["line","rectangular hyperbola","parallel lines"])
    split = false
    if bezptsᵣ != []
        if maximum(abs.(bezptsᵣ)) > def_rational_bezpt_dist 
            split = true
        end
    end
    if (insimplex(saddlepoint(bezpts[end,:],atol=atol)) && !linear)
        split = true
    end
    if split
        bezptsᵤ = [split_bezsurf(b,atol=atol) for b=split_bezsurf₁(bezpts)] |> flatten |> collect
        return sum([two₋intersects_area₋volume(b,quantity,atol=atol) for b=bezptsᵤ])
    end

    # No intersections but some of the coefficients are less or greater than 0.
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
    quad_area₋volume(bezpts,quantity;atol)

Calculate the area of the shadow of a quadric or the volume beneath the quadratic.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `quantity::String`: the quantity to calculate ("area" or "volume").
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

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
        triangle = bezpts[1:end-1,corner_indices]
    end
    sum([two₋intersects_area₋volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])    
end

face_ind = [[1,2,3],[3,4,1],[4,1,2],[2,3,4]]
@doc """
    get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,simplicesᵢ,fatten)

Calculate the interval Bezier points for all sheets.

# Arguments
- `index::Int`: the index of the simplex in `simplicesᵢ`.
- `mesh::PyObject`: a triangulation of the irreducible Brillouin zone.
- `ext_mesh::PyObject`: a triangulation of the region within and around
    the IBZ.
- `sym₋unique::AbstractVector{<:Real}`: the index of the eigenvalues for each point
    in the `mesh`.
- `eigenvals::AbstractMatrix{<:Real}`: a matrix of eigenvalues for the symmetrically
    distinc points as columns of a matrix.
- `simplicesᵢ::Vector`: the simplices of `mesh` that do not include the box points.
- `fatten::Real`: scale the interval coefficients by this amount.
- `num_near_neigh::Int`: the number of neighbors included.
- `sigma::Int`: the number of sheets summed and then interpolated, if any.

# Returns
- `inter_bezpts::Vector{Matrix{Float64}}`: the interval Bezier points
    for each sheet.

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
function get_intercoeffs(index::Int, mesh::PyObject, ext_mesh::PyObject,
        sym₋unique::AbstractVector{<:Real}, eigenvals::AbstractMatrix{<:Real},
        simplicesᵢ::Vector,
        fatten::Real=def_fatten,
        num_near_neigh::Int=def_num_near_neigh; sigma::Real=0,
        epm::Union{Nothing,epm₋model2D,epm₋model}=nothing,
        neighbor_method::Int=def_neighbor_method,
        num_neighbors::Union{Nothing,Integer}=nothing)
     
    simplexᵢ = simplicesᵢ[index]
    simplex = Matrix(mesh.points[simplexᵢ,:]')
    dim = size(simplex,2)-1 
    neighborsᵢ = reduce(vcat,[get_neighbors(s,ext_mesh,num_near_neigh) for s=simplexᵢ]) |> unique
    neighborsᵢ = filter(x -> !(x in simplexᵢ),neighborsᵢ)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end

    if length(neighborsᵢ) < num_neighbors num_neighbors = length(neighborsᵢ) end

    if neighbor_method == 1
        # Select neighbors that are closest to the triangle.
        neighbors = ext_mesh.points[neighborsᵢ,:]'
        dist = [minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:dim+1]) for i=neighborsᵢ]
        neighborsᵢ = neighborsᵢ[sortperm(dist)][1:num_neighbors]

    elseif neighbor_method == 2
        # Select neighbors that surround the triangle and are close to the triangle.
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
    # eigvals = zeros(size(eigenvals,2),15)
    if neighbor_method == 3
        n = def_inside_neighbors_divs # Number of points for the uniform sampling of the triangle
        b = sample_simplex(dim,n)
        b = b[:,setdiff(1:size(b,2),[1,n+1,size(b,2)])]
        eigvals = eval_epm(barytocart(b,simplex),epm)
    else
        b = carttobary(ext_mesh.points[neighborsᵢ,:]',simplex)
    end

    # Constrained least-squares
    if dim == 2
        M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3]],b,dims=1)'
    else
        M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3],x[1]*x[4],x[2]*x[4],x[3]*x[4]],b,dims=1)'
    end

    # Unconstrained least-squares
    # if dim == 2
    #     b = [[1 0 0; 0 1 0; 0 0 1] b]
    #     M = mapslices(x->[x[1]^2,2*x[1]*x[2],x[2]^2,2*x[1]*x[3],2*x[2]*x[3],x[3]^2],b,dims=1)'
    # else
    #     b = [[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1] b]
    #     M = mapslices(x->[x[1]^2,2*x[1]*x[2],x[2]^2,2*x[1]*x[3],2*x[2]*x[3],x[3]^2,
    #         2*x[1]*x[4],2*x[2]*x[4],2*x[3]*x[4],x[4]^2],b,dims=1)' 
    # end

    # Minimum distance from the edges of the triangle.
    # if dim == 2
    #     W = diagm([minimum([lineseg₋pt_dist(ext_mesh.points[i,:],simplex[:,s]) for s=[[1,2],[2,3],[3,1]]])
    #         for i=neighborsᵢ])
    # else 
    #     W = diagm([minimum([ptface_mindist(ext_mesh.points[i,:],simplex[:,s]) for s=face_ind])
    #         for i=neighborsᵢ])
    # end
 
    # Distance from the center of the triangle.
    # W = diagm([norm(ext_mesh.points[i,:] - mean(simplex,dims=2)) for i=neighborsᵢ])
    
    # Shortest distance from one of the corners of the triangle.
    # W = diagm([1/minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:dim+1])^2 for i=neighborsᵢ])
    # W=I

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
        # Constrained least-squares
        Z = fᵢ - (b.^2)'*q

        # Unconstrained least-squares
        # fᵢ = [q; fᵢ];
        # Z = fᵢ
         
        c = M\Z
        # Weighted least squares
        # c = pinv(M)*Z        
        # c = inv(M'*W*M)*M'*W*Z
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
            # Constrained least-squares
            bezcoeffs[sheet] = scoeffs

            # Unconstrained least-squares
            # bezcoeffs[sheet] = c
        else
            # Constrained least-squares
            bezcoeffs[1] = scoeffs

            # Unconstrained least-squares
            # bezcoeffs[1] = c
        end
        # Constrained least-squares        
        qᵢ = [eval_poly(b[:,i],scoeffs,dim,2) for i=1:size(b,2)]

        # Unconstrained least-squares
        # qᵢ = [eval_poly(b[:,i],c,dim,2) for i=1:size(b,2)]

        δᵢ = fᵢ - qᵢ; 
        ϵ = Matrix(reduce(hcat,[(1/dot(M[i,:],M[i,:])*δᵢ[i])*M[i,:] for i=1:length(δᵢ)])')
        ϵ = [minimum(ϵ,dims=1); maximum(ϵ,dims=1)].*fatten
        # ϵ = [-abs.(mean(ϵ,dims=1)); abs.(mean(ϵ,dims=1))].*fatten
        intercoeffs = [c';c'] .+ ϵ
    
        # Constrained least-squares
        if dim == 2
            c1,c2,c3 = [intercoeffs[:,i] for i=1:3]
            intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3]])
        else
            c1,c2,c3,c4,c5,c6 = [intercoeffs[:,i] for i=1:6]
            intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3],c4,c5,c6,[q4,q4]])
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
    Calculate the Fermi level of a representation of the band structure.

"""
function calc_fl(epm::Union{epm₋model,epm₋model2D},ebs::bandstructure; 
    num_slices::Int = 10, window::Union{Nothing,Vector{<:Real}}=ebs.fermilevel_interval, 
        ctype="mean", fermi_area::Real=epm.fermiarea/length(epm.pointgroup))

    if ctype == "mean"
        cfun = mean 
    elseif ctype == "min"
        cfun = minimum 
    elseif ctype == "max"
        cfun = maximum
    else 
        error("Invalid ctype.")
    end
    
    dim = size(epm.recip_latvecs,1)
    simplex_bpts = sample_simplex(dim,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    
    ibz_area = epm.ibz.volume
    maxsheet = round(Int,epm.electrons/2) + 2
    window = nothing

    estart = if dim == 2 5 else 9 end
    if window == nothing || window == [0,0]
        E₁ = minimum(ebs.eigenvals[1,estart:end])
        E₂ = maximum(ebs.eigenvals[maxsheet,estart:end])
    else
       E₁,E₂ = window
    end
    # Make sure the window contains the approx. Fermi level.
    dE = 2*abs(E₂ - E₁)
    if dim == 2
        f₁ = sum([quad_area₋volume([simplex_pts[tri]; [minimum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₁],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
    else
        f₁ = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- E₁,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area
    end
    iters₁ = 0
    while f₁ > 0
        iters₁ += 1; E₁ -= dE; dE *= 2
        if iters₁ > def_fl_max_iters || dE == 0
            E₁ = minimum(ebs.eigenvals[1,:5:end])
        end
        if dim == 2
            f₁ = sum([quad_area₋volume([simplex_pts[tri]; [minimum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₁],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
        else
            f₁ = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- E₁,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area
        end
    end

    dE = 2*abs(E₂ - E₁)
    if dim == 2
        f₂ = sum([quad_area₋volume([simplex_pts[tri]; [maximum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₂],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
    else
        f₂ = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- E₂,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area
    end
    iters₂ = 0
    while f₂ < 0
        iters₂ += 1; E₂ += dE; dE *= 2
        if iters₂ > def_fl_max_iters || dE == 0
            E₂ = maximum(ebs.eigenvals[maxsheet,5:end])
        end
        if dim == 2
            f₂ = sum([quad_area₋volume([simplex_pts[tri]; [maximum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₂],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
        else
            f₂ = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- E₂,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area
        end
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

        if ctype == "mean"
            if dim == 2
                f = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_bezcoeffs[tri][sheet]' .- E],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
            else
                f = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- E,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area
            end 
        else
            if dim == 2
                f = sum([quad_area₋volume([simplex_pts[tri]; [cfun(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area 
            else
                f = sum([simpson3D(vec(cfun(ebs.mesh_intcoeffs[i][j],dims=1)) .- E,simplices[i],num_slices,"area") for i=1:length(ebs.simplicesᵢ) for j=1:epm.sheets]) - fermi_area 
            end
        end

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
    E
end

@doc """
    calc_flbe!(epm,ebs;num_slices)

Calculate the Fermi level and band energy for a given rep. of the band struct.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ebs::bandstructure`: the band structure container 

# Returns
- `ebs::bandstructure`: update values within container for the band energy error,
    Fermi area error, Fermi level interval, Fermi area interval, band energy
    interval, and the partially occupied sheets.
"""
function calc_flbe!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure;
    num_slices=10)
     
    dim = size(epm.recip_latvecs,1)
    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(dim,2) 
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The areas of the triangles in the mesh
    simplex_sizes = [simplex_size(s) for s=simplices]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
     
    # The number of point operators
    npg = length(epm.pointgroup)
     
    # The "true" Fermi level for the given approximation of the band structure
    fl = calc_fl(epm,ebs,fermi_area=epm.fermiarea/npg,ctype="mean", num_slices=num_slices)
    # The larger Fermi level computed with the upper limit of approximation intervals
    fl₁ = calc_fl(epm,ebs,fermi_area=epm.fermiarea/npg,ctype = "max", num_slices=num_slices)
    # The smaller Fermi level computed with the lower limit of approximation intervals
    fl₀ = calc_fl(epm,ebs,fermi_area=epm.fermiarea/npg,ctype = "min", num_slices=num_slices)

    # The "true" Fermi area for each quadratic triangle (triangle and sheet) for the
    # given approximation of the bandstructure
    if dim == 2
        mesh_fa = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_bezcoeffs[tri][sheet] .- fl)']
            ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    else
        mesh_fa = [[simpson3D(ebs.mesh_bezcoeffs[i][j] .- fl,simplices[i],num_slices,"area") 
        for j=1:epm.sheets] for i=1:length(ebs.simplicesᵢ)]
    end

    # The smaller Fermi area for each quadratic triangle using the upper limit of the approximation
    # intervals with the lower limit of the Fermi level interval
    # Minimum
    # if dim == 2
    #     mesh_fa₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl)']
    #         ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    #     mesh_fa₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl)']
    #         ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    # else
    #     mesh_fa₁ = [[simpson3D(ebs.mesh_intcoeffs[i][j][1,:] .- fl,simplices[i],num_slices,"area") 
    #     for j=1:epm.sheets] for i=1:length(ebs.simplicesᵢ)]
    #     mesh_fa₀ = [[simpson3D(ebs.mesh_intcoeffs[i][j][2,:] .- fl,simplices[i],num_slices,"area") 
    #     for j=1:epm.sheets] for i=1:length(ebs.simplicesᵢ)]
    # end

    # The larger Fermi area for each quadratic triangle using the lower limit of the approximation
    # intervals with the upper limit of the Fermi level interval
    # Maximum
    if dim === 2
        mesh_fa₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)']
            ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
        mesh_fa₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)']
            ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    else
        mesh_fa₁ = [[simpson3D(ebs.mesh_intcoeffs[i][j][1,:] .- fl₁,simplices[i],num_slices,"area") 
        for j=1:epm.sheets] for i=1:length(ebs.simplicesᵢ)]
        mesh_fa₀ = [[simpson3D(ebs.mesh_intcoeffs[i][j][2,:] .- fl₀,simplices[i],num_slices,"area") 
        for j=1:epm.sheets] for i=1:length(ebs.simplicesᵢ)]
    end

    # The smaller and larger Fermi areas or the limits of the Fermi area interval
    fa₀,fa₁ = sum(sum(mesh_fa₀)),sum(sum(mesh_fa₁))
    # The "true" band energy for the given approximation of the band structure
    if dim == 2
        be = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_bezcoeffs[tri][sheet]' .- fl], "volume") 
        for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])
    else
        be = sum([simpson3D(ebs.mesh_bezcoeffs[i][j] .- fl,simplices[i],num_slices,"volume") 
        for j=1:epm.sheets for i=1:length(ebs.simplicesᵢ)]) 
    end

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
    ) for sheet=1:epm.sheets] for tri = 1:length(ebs.simplicesᵢ)]

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
    ) for sheet=1:epm.sheets] for tri = 1:length(ebs.simplicesᵢ)]
    
    # Another knob to make partial band energy errors smaller
    # partial_occ = true_partial_occ 
    
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
        end) for i=1:length(ebs.simplicesᵢ)]

    # Calculate the "sigma" coefficients for the "true" occupations of the sheets. The true sigma 
    # coefficients and intervals and the regular coefficients and intervals are only different if the
    # occupations are different
    true_sigma_intcoeffs = [
        (if true_sigmas[i] == nothing
            [[zeros(2,nterms)],[zeros(1,nterms)]]
        else
            get_intercoeffs(i,ebs.mesh,ebs.ext_mesh,ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,sigma=true_sigmas[i],epm=epm,
            neighbor_method = ebs.neighbor_method)
        end) for i=1:length(ebs.simplicesᵢ)]
    
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
    if dim == 2
        partial_be = [[fl*mesh_fa[tri][sheet] + quad_area₋volume([simplex_pts[tri]; (ebs.mesh_bezcoeffs[tri][sheet] .- fl)']
            ,"volume") for sheet=true_partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be = [e == [] ? 0 : sum(e) for e = partial_be]
    else
        partial_be = [[fl*mesh_fa[tri][sheet] + simpson3D(ebs.mesh_bezcoeffs[tri][sheet] .- fl,simplices[tri],num_slices,"volume") for sheet=true_partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be = [e == [] ? 0 : sum(e) for e = partial_be]
    end

    # The lower limit of the contribution to the band energy from the partially occupied sheets
    # obtained using the regular occupations, the lower limit of the interval coefficients,
    # and the upper limit of the Fermi level interval

    # Minimum
    # if dim == 2
    #     partial_be₀ = [[fl*mesh_fa[tri][sheet] + quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl)'],"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
    #     partial_be₀ = [e == [] ? 0 : sum(e) for e = partial_be₀]

    #     partial_be₁ = [[fl*mesh_fa[tri][sheet] + quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl)'],"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
    #     partial_be₁ = [e == [] ? 0 : sum(e) for e = partial_be₁]
    # else
    #     partial_be₀ = [[fl*mesh_fa[tri][sheet] + simpson3D(ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl,simplices[tri],num_slices,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
    #     partial_be₀ = [e == [] ? 0 : sum(e) for e = partial_be₀]

    #     partial_be₁ = [[fl*mesh_fa[tri][sheet] + simpson3D(ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl,simplices[tri],num_slices,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
    #     partial_be₁ = [e == [] ? 0 : sum(e) for e = partial_be₁]
    # end

    # Maximum
    if dim == 2
        partial_be₀ = [[fl₀*mesh_fa₀[tri][sheet] + quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)'],"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be₀ = [e == [] ? 0 : sum(e) for e = partial_be₀]
    
        partial_be₁ = [[fl₁*mesh_fa₁[tri][sheet] + quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)'],"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be₁ = [e == [] ? 0 : sum(e) for e = partial_be₁]
    else
        partial_be₀ = [[fl₀*mesh_fa₀[tri][sheet] + simpson3D(ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀,simplices[tri],num_slices,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be₀ = [e == [] ? 0 : sum(e) for e = partial_be₀]

        partial_be₁ = [[fl₁*mesh_fa₁[tri][sheet] + simpson3D(ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁,simplices[tri],num_slices,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
        partial_be₁ = [e == [] ? 0 : sum(e) for e = partial_be₁]
    end

    # The upper limit of the contribution to the band energy from the partially occupied sheets
    # obtained using the regular occupations, the upper limit of the interval coefficients,
    # and the lower limit of the Fermi level interval

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
"""
function refine_mesh!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure)
     
    spatial = pyimport("scipy.spatial")
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    err_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.target_accuracy
    faerr_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.fermiarea_eps
     
    n = def_min_split_triangles
    # Refine the tile with the most error
    if ebs.refine_method == 1
        splitpos = sortperm(abs.(ebs.bandenergy_errors),rev=true)
        if length(splitpos) > n splitpos = splitpos[1:n] end

    # Refine the tiles with too much error (given the tiles' sizes).
    elseif ebs.refine_method == 2
        splitpos = filter(x -> x>0,[abs(ebs.bandenergy_errors[i]) > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])

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
        if length(order) > n
            splitpos = order[1:round(Int,length(order)*def_frac_refined)]
        else
            splitpos = order
        end        
    else
        ArgumentError("The refinement method has to be and integer equal to 1,..., 6.")
    end
    frac_split = length(splitpos)/length(ebs.bandenergy_errors)

    println("Number of split triangles: ", length(splitpos))
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
        ArgumentError("The sample method for refinement has to be an integer with a value of 1 or 2.")
    end

    # Remove duplicates from the new mesh points.
    new_meshpts = unique_points(new_meshpts,rtol=ebs.rtol,atol=ebs.atol)
    new_eigvals = eval_epm(new_meshpts,epm,rtol=ebs.rtol,atol=ebs.atol)
    ebs.eigenvals = [ebs.eigenvals new_eigvals]

    @show size(new_meshpts,2)

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
"""
function get_tolerances(epm,ebs)

    stepsize = (maximum(ebs.eigenvals[:,5:end])-minimum(ebs.eigenvals[:,5:end]))/1e4
    numsteps = 4
    es = collect(-numsteps/2*stepsize:stepsize:numsteps/2*stepsize) .+ ebs.fermilevel

    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(2,2) 
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]

    npg = length(epm.pointgroup)
    bens = []
    fas = []

    npg = length(epm.pointgroup)
    for fl=es
        be = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_bezcoeffs[tri][sheet]' .- fl], "volume") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])
        fa = npg*sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_bezcoeffs[tri][sheet]' .- fl], "area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])
        # be = 2*npg*(be + fl*epm.fermiarea/npg)
        # Remove translation of band energy because numbers close to zero throw off the derivative
        be = 2*npg*be
        push!(bens,be); push!(fas,fa)
    end
    dbs = [(-bens[i+2] + 8*bens[i+1] - 8*bens[i-1] + bens[i-2])/(es[i+2]-es[i-2]) for i=3:length(bens)-2]
    das = [(-fas[i+2] + 8*fas[i+1] - 8*fas[i-1] + fas[i-2])/(es[i+2]-es[i-2]) for i=3:length(fas)-2];

    db = maximum(abs.(das))
    da = maximum(abs.(dbs))

    dltol = ebs.target_accuracy/db
    datol = da*dltol
    
    db,da,dltol,datol
end

@doc """
    quadratic_method!(epm,ebs,init_msize,num_near_neigh,fermiarea_eps,target_accuracy,
        fermilevel_method,refine_method,sample_method,rtol,atol,uniform)

Calculate the band energy using uniform or adaptive quadratic integation.
"""
function quadratic_method!(epm::Union{epm₋model2D,epm₋model};
    init_msize::Int=def_init_msize, num_near_neigh::Int=def_num_near_neigh,
    num_neighbors::Union{Nothing,Int}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Int=def_fermilevel_method, 
    refine_method::Int=def_refine_method,
    sample_method::Int=def_sample_method, 
    neighbor_method::Int=def_neighbor_method, 
    fatten::Real=def_fatten,
    rtol::Real=def_rtol,
    atol::Real=def_atol,
    uniform::Bool=def_uniform)

    dim = size(epm.recip_latvecs,1)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end
    
    ebs = init_bandstructure(epm,init_msize=init_msize, num_near_neigh=num_near_neigh,
        num_neighbors=num_neighbors,fermiarea_eps=fermiarea_eps, 
        target_accuracy=target_accuracy, fermilevel_method=fermilevel_method, refine_method=refine_method, sample_method=sample_method, 
        neighbor_method=neighbor_method, fatten=fatten, rtol=rtol, atol=atol)
    calc_flbe!(epm,ebs)

    if uniform return ebs end
    db,da,dltol,datol = get_tolerances(epm,ebs)
    ebs.fermiarea_eps = datol; 
    counter = 0
    diff_BE = 1e9
    prev_BE = ebs.bandenergy
    tmp = []
    sd = 3 # rounding parameter for print statements
    # while abs(sum(ebs.bandenergy_errors)) > ebs.target_accuracy
    while diff_BE > ebs.target_accuracy
    # while db/da*diff(ebs.fermiarea_interval)[1]/2 > ebs.target_accuracy
        println("Number of triangles: ", length(ebs.simplicesᵢ)) 
        if counter != 1
            db,da,dltol,datol = get_tolerances(epm,ebs)
            tmp = round.([db,da,dltol,datol],sigdigits=6) 
            # ebs.fermiarea_eps = datol/10;
        end
        counter += 1; refine_mesh!(epm,ebs); calc_flbe!(epm,ebs);
        if counter > max_refine_steps
            @warn "Failed to calculate the band energy to within the desired accuracy $(ebs.target_accuracy) after 100 iterations."
            break
        end
        diff_BE = abs(ebs.bandenergy - prev_BE)
        prev_BE = ebs.bandenergy 
        ϵᵦ = abs(ebs.bandenergy - epm.bandenergy) 
        ϵₗ = abs(ebs.fermilevel - epm.fermilevel)
        println("\nTrue errors")
        println("ϵᵦ: ", round(ϵᵦ,sigdigits=sd))
        println("ϵₗ: ", round(ϵₗ,sigdigits=sd))
        # println("BE IS: ", diff(ebs.bandenergy_interval)[1]/tb)
        println("Estimated band energy error")
        println("δB/ϵᵦ: ", round(diff_BE/ϵᵦ,sigdigits=sd))
        println("ΔB/ϵᵦ: ", round(sum(ebs.bandenergy_errors)/ϵᵦ,sigdigits=sd))
        println("dB/dL*ΔLₜ/ϵᵦ: ", round(db*diff(ebs.fermilevel_interval)[1]/2/ϵᵦ,sigdigits=sd))
        println("dB/dA*ΔAₜ/ϵᵦ: ", round(db/da*diff(ebs.fermiarea_interval)[1]/2/ϵᵦ,sigdigits=sd))
        
        println("Estimated Fermi level error")
        println("ΔL/ϵₗ: ", round(diff(ebs.fermilevel_interval)[1]/2/ϵₗ,sigdigits=sd))
        println("Estimated Fermi area errors")
        println("ΔA: ", round(diff(ebs.fermiarea_interval)[1]/2,sigdigits=sd))
        println("dA/dL*ΔLₜ: ", round(da*diff(ebs.fermilevel_interval)[1]/2,sigdigits=sd))
        
        println("Derivatives and tolerances")
        println("dB/dL dA/dL dB/dA ΔAₜ ΔLₜ = ",round.([tmp[1],tmp[2],db/da,tmp[3],tmp[4]],sigdigits=sd))
    end
    ebs
end

"""
    truebe(ebs,epm,ndivs)

Calculate (roughly) the true band energy error for each quadratic triangle.
"""
function truebe(ebs,epm,ndivs;num_cores=1,triangles=nothing)
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
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.

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

Calculate the area of a quadratic where it is less than zero between (0,1).

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

Integrate the area below a list of values within an interval.

# Arguments
- `y::AbstractVector{<:Real}`: a list of values of the function being integrated.
- `int_len::Real`: the length of the interval over which the funcion is integrated.

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
using Pebsi.QuadraticIntegration: linept_dist
line = [0 1; 0 0]
pt = [0,2]
# output
2.0
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

function tetface_areas(tet)
    areas = zeros(4)
    for (i,j)=enumerate(face_ind)
        areas[i] = norm(cross(tet[:,j[2]] - tet[:,j[1]], tet[:,j[3]] - tet[:,j[1]]))/2
    end
    areas
end

@doc """
    simpson3D(coeffs,tetrahedron,num_slices,quantity;values,split,atol,rtol)

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
function simpson3D(coeffs,tetrahedron,num_slices,quantity;values=false,gauss=true,
    atol=def_atol,split=true,corner=nothing)
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

function quadslice_tanpt(coeffs; atol=def_atol,rtol=def_rtol)
     
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

end # module
