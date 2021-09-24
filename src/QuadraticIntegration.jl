module QuadraticIntegration

using SymmetryReduceBZ.Utilities: unique_points, shoelace
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ..Polynomials: eval_poly,getpoly_coeffs,getbez_pts₋wts,eval_bezcurve,
    conicsection
using ..EPMs: eval_epm, RytoeV, epm₋model, epm₋model2D
using ..Mesh: get_neighbors,notbox_simplices,get_cvpts,ibz_init₋mesh, 
    get_extmesh, choose_neighbors
using ..Geometry: order_vertices!,simplex_size,insimplex,barytocart,carttobary,
    sample_simplex,lineseg₋pt_dist
using ..Simpson: bezcurve_intersects
using ..Defaults

using QHull: chull,Chull
using LinearAlgebra: cross,det,norm,dot,I,diagm,pinv
using Statistics: mean
using Base.Iterators: flatten
using SparseArrays: findnz
using PyCall: PyObject, pyimport
using Distributed: pmap

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
- `approx_fermilevel::Real`: the approximate Fermi level. This is the midpoint of
    the Fermi level interval.
- `approx_bandenergy::Real`: the approximate band energy. This is the midpoint of
    the band energy interval.
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
    approx_fermilevel::Real
    approx_bandenergy::Real
    fermiarea_interval::AbstractVector{<:Real}
    fermilevel_interval::AbstractVector{<:Real}
    bandenergy_interval::AbstractVector{<:Real}
    fermilevel::Real
    bandenergy::Real
    partially_occupied::Vector{Vector{Int64}}
    bandenergy_errors::Vector{<:Real}
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
    epm::Union{epm₋model,epm₋model2D};
    init_msize::Int=def_init_msize,
    num_near_neigh::Int=def_num_near_neigh,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Int=def_fermilevel_method,
    refine_method::Int=def_refine_method,
    sample_method::Int=def_sample_method,
    neighbor_method::Int=def_neighbor_method,
    fatten::Real=def_fatten,
    inside::Bool=def_inside,
    rtol::Real=def_rtol,
    atol::Real=def_atol)

    if inside model = epm else model = nothing end

    mesh = ibz_init₋mesh(epm.ibz,init_msize;rtol=rtol,atol=atol)
    mesh,ext_mesh,sym₋unique = get_extmesh(epm.ibz,mesh,epm.pointgroup,
        epm.recip_latvecs,num_near_neigh; rtol=rtol,atol=atol)
    simplicesᵢ = notbox_simplices(mesh)
 
    uniqueᵢ = sort(unique(sym₋unique))[2:end]
    # eigenvals = zeros(Float64,epm.sheets,size(mesh.points,1))
    eigenvals = zeros(Float64,epm.sheets,4+length(uniqueᵢ))
    for i=uniqueᵢ
        eigenvals[:,i] = eval_epm(mesh.points[i,:], epm, rtol=rtol, atol=atol)
    end
    
    mesh_intcoeffs = [get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,
        simplicesᵢ,fatten,num_near_neigh,epm=model,neighbor_method=neighbor_method) for index=1:length(simplicesᵢ)];
    
    partially_occupied = [zeros(Int,epm.sheets) for _=1:length(simplicesᵢ)]
    bandenergy_errors = zeros(length(simplicesᵢ))
    fermiarea_errors = zeros(length(simplicesᵢ))
    
    approx_fermilevel=0
    approx_bandenergy=0
    fermiarea_interval=[0,0]
    fermilevel_interval=[0,0]
    bandenergy_interval=[0,0]
    fermilevel=0
    bandenergy=0
        
    bandstructure(
        init_msize,
        num_near_neigh,
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
        approx_fermilevel,
        approx_bandenergy,
        fermiarea_interval,
        fermilevel_interval,
        bandenergy_interval,
        fermilevel,
        bandenergy,
        partially_occupied,
        bandenergy_errors,
        fermiarea_errors)
end   

@doc """
    simpson(interval_len,vals)

Calculate the integral of a univariate function with the composite Simpson's method

# Arguments
- `interval_len::Real`: the length of the inteval the functios is integrated over.
- `vals::AbstractVector{<:Real}`: the value of the function on a uniform, closed 
    grid over the interval.

# Returns
- `::Real`: the approximate integral of the function over the iterval.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: simpson
num_intervals = 20
f(x) = x^5 - x^4 - 2*x^3
vals = map(x->f(x),collect(0:1/(2*num_intervals):1))
interval_len = 1
answer = (-8/15)
abs(simpson(interval_len,vals) - answer)
# output
7.812500002479794e-8
```
"""
function simpson(interval_len::Real,vals::AbstractVector{<:Real})::Real
    num_intervals = Int((length(vals) - 1)/2)
    simp_wts = ones(Int,2*num_intervals+1)
    simp_wts[2:2:end-1] .= 4
    simp_wts[3:2:end-2] .= 2 
    interval_len/(6*num_intervals)*dot(simp_wts,vals)
end

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
    atol::Real=def_atol)::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = bezcurve_intersects(edge_bezpts[end,:];atol=atol)
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
- `atol::Real=1e-9`: absolute tolerance.

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
    allpts::AbstractArray=[]; atol::Real=def_atol)::AbstractArray
    spatial = pyimport("scipy.spatial")
    dim = 2
    deg = 2
    triangle = bezpts[1:2,corner_indices]
    coeffs = bezpts[end,:]
    pts = bezpts[1:2,:]
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
import Pebsi.QuadraticIntegration: split_triangle
bezpts = [-0.09385488270304788 0.12248162346376468 0.3388181296305772 0.09890589198180941 0.315242398148622 0.2916666666666667; 0.9061451172969521 0.7836634938331875 0.6611818703694228 0.6266836697595872 0.5042020462958225 0.34722222222222227; 0.0 7.949933953535975 3.9968028886505635e-15 8.042737134030771 -5.792491135426262 -11.720219017094017]
split_bezsurf₁(bezpts)
# output
2-element Array{Array{Float64,2},1}:
 [0.1291676795676943 0.23399290459913574 … 0.12248162346376468 -0.09385488270304788; 0.5828106204960847 0.6219962454327537 … 0.7836634938331875 0.9061451172969521; -5.329070518200751e-15 -4.5330445462060594e-15 … 7.9499339535359725 0.0]
 [0.1291676795676943 0.2104171731171805 … 0.315242398148622 0.3388181296305772; 0.5828106204960847 0.46501642135915344 … 0.5042020462958225 0.6611818703694228; -5.329070518200751e-15 -3.39004820851129 … -5.792491135426261 -1.1479627341393213e-15]
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
0.4426972170733675
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

    # Calculate the bezier curve and weights make sure the curve passes through
    # the triangle
    triangle = bezpts[1:2,corner_indices]
    coeffs = bezpts[end,:]
    intersects = simplex_intersects(bezpts,atol=atol)

    # No intersections
    if intersects == [[],[],[]]
        # Case where the sheet is completely above or below 0.    
        if all(bezpts[end,:] .< 0) && !all(isapprox.(bezpts[end,:],0))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
            return areaₒᵣvolume
        end
        if all(bezpts[end,:] .> 0) && !all(isapprox.(bezpts[end,:],0))
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
        if maximum(abs.(bezptsᵣ)) > 1e6 
            split = true
        end
    elseif (insimplex(saddlepoint(bezpts[end,:],atol=atol)) && !linear)
        split = true
    else
        nothing
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
quad_area₋volume(bezpts,"area")
# output
0.869605101106897
```
"""
function quad_area₋volume(bezpts::AbstractMatrix{<:Real},
        quantity::String;atol::Real=def_atol)::Real
    sum([two₋intersects_area₋volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])    
end

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
- `simplicesᵢ::Vector{Vector{Int64}}`: the simplices of `mesh` that do not
    include the box points.
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
import Pebsi.QuadraticIntegration: get_inter₋bezpts

n = 10
mesh = ibz_init₋mesh(m2ibz,n)
simplicesᵢ = notbox_simplices(mesh)

num_near_neigh = 2
ext_mesh,sym₋unique = get_extmesh(m2ibz,mesh,m2pointgroup,m2recip_latvecs,num_near_neigh)

sheets = 7
energy_conv = 1
eigenvals = zeros(sheets,size(mesh.points,1))
for i = sort(unique(sym₋unique))[2:end]
    eigenvals[:,i] = eval_epm(mesh.points[i,:],m2recip_latvecs,m2rules,m2cutoff,sheets,energy_conv)
end

index = 1
get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,simplicesᵢ)
# output
7-element Vector{Matrix{Float64}}:
 [-0.4170406590890757 -0.44894253681741786 … -0.418185036063509 -0.4087992707500061; -0.4170406590890757 -0.4130115291504489 … -0.38325424751903675 -0.4087992707500061]
 [-0.09968473377219263 -0.10467222688790542 … -0.16182723345176916 -0.11471023344428993; -0.09968473377219263 -0.03966774615259443 … -0.09724196302121388 -0.11471023344428993]
 [0.06333883794674595 0.06176891894277915 … 0.05053770503975599 0.059530423104755405; 0.06333883794674595 0.07433130559535622 … 0.06321666534916602 0.059530423104755405]
 [0.9336184268894858 0.8965079932976808 … 0.9422896105253507 0.9616264337394995; 0.9336184268894858 0.9442386828910152 … 0.9986202705442639 0.9616264337394995]
 [1.0370385907264408 0.98617538886686 … 1.0192740316847344 1.025752774169218; 1.0370385907264408 1.0340184952232654 … 1.0650238198456579 1.025752774169218]
 [1.243798381547987 1.1209957076784376 … 1.2392094226656643 1.2828198059158602; 1.243798381547987 1.255588675708819 … 1.3708953013582792 1.2828198059158602]
 [1.7629457567764115 1.7492156915207968 … 1.586750383315745 1.7117209463142664; 1.7629457567764115 1.9545533797734849 … 1.7735112086457399 1.7117209463142664]
```
"""
function get_intercoeffs(index::Int,mesh::PyObject,ext_mesh::PyObject,
        sym₋unique::AbstractVector{<:Real},eigenvals::AbstractMatrix{<:Real},
        simplicesᵢ::Vector{Vector{Int64}},fatten::Real=def_fatten,
        num_near_neigh::Int=def_num_near_neigh; sigma::Real=0,
        epm::Union{Nothing,epm₋model2D}=nothing,
        neighbor_method::Int=def_neighbor_method,
        num_neighbors::Int=def_num_neighbors)::Vector{Matrix{Float64}}

    simplexᵢ = simplicesᵢ[index]
    simplex = Matrix(mesh.points[simplexᵢ,:]')
    neighborsᵢ = reduce(vcat,[get_neighbors(s,ext_mesh,num_near_neigh) for s=simplexᵢ]) |> unique
    neighborsᵢ = filter(x -> !(x in simplexᵢ),neighborsᵢ)

    if length(neighborsᵢ) < num_neighbors num_neighbors = length(neighborsᵢ) end

    if neighbor_method == 1
        # Select neighbors that are closest to the triangle.
        neighbors = ext_mesh.points[neighborsᵢ,:]'
        dist = [minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:3]) for i=neighborsᵢ]
        neighborsᵢ = neighborsᵢ[sortperm(dist)][1:num_neighbors]

    elseif neighbor_method == 2
        # Select neighbors that surround the triangle and are close to the triangle.
        neighbors = Matrix(ext_mesh.points[neighborsᵢ,:]')
        neighborsᵢ = choose_neighbors(simplex,neighborsᵢ,neighbors; num_neighbors=num_neighbors)

    # Neighbors are taken from a uniform grid within the triangle.
    elseif neighbor_method == 3
        neighborsᵢ = []
        if epm == nothing
            error("Must provide an EPM when computing neighbors within the triangle.")
        end
    else
        error("Only 1, 2, and 3 are valid values of the flag for the method of selecting neighbors.")
    end

    eigvals = zeros(size(eigenvals,2),15)
    if neighbor_method == 3
        n = def_inside_neighbors_divs # Number of points for the uniform sampling of the triangle
        b = sample_simplex(2,n)
        b = b[:,setdiff(1:length(b),[1,n+1,length(b)])]
        eigvals = eval_epm(barytocart(b,simplex),epm)
    else
        b = carttobary(ext_mesh.points[neighborsᵢ,:]',simplex)
    end
    M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3]],b,dims=1)'
    Dᵢ = mapslices(x -> sum((x/2).^2),M,dims=2)
        
    # Minimum distance from the edges of the triangle.
    # W = diagm([minimum([lineseg₋pt_dist(simplex[:,s],ext_mesh.points[i,:]) for s=[[1,2],[2,3],[3,1]]])
    #     for i=neighborsᵢ])
 
    # Distance from the center of the triangle.
    # W = diagm([norm(ext_mesh.points[i,:] - mean(simplex,dims=2)) for i=neighborsᵢ])
    
    # Shortest distance from one of the corners of the triangle.
    # W = diagm([1/minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:3])^2 for i=neighborsᵢ])
    # W=I

    if sigma == 0
        inter_bezcoeffs = [zeros(2,size(eigenvals,1)) for i=1:size(eigenvals,1)]
    else
        inter_bezcoeffs = [zeros(2,size(eigenvals,1)) for i=1:1]
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
        Z = fᵢ - (b.^2)'*q
          
        # Weighted least squares
        c = M\Z
        # c = pinv(M)*Z        
        # c = inv(M'*W*M)*M'*W*Z
        c1,c2,c3 = c
        q1,q2,q3 = q
        qᵢ = [eval_poly(b[:,i],[q1,c1,q2,c2,c3,q3],2,2) for i=1:size(b,2)]
        δᵢ = fᵢ - qᵢ;
        ϵ = δᵢ./(2Dᵢ).*M
        ϵ = [minimum(ϵ,dims=1);maximum(ϵ,dims=1)]

        # Scale the interval to make errors more accurate.
        c = [c[i] .+ ϵ[:,i]*fatten for i=1:3]

        c1,c2,c3 = c
        intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3]])
        if sigma == 0
            inter_bezcoeffs[sheet] = intercoeffs
        else
            inter_bezcoeffs[1] = intercoeffs
            break
        end
    end

    Vector{Matrix{Float64}}(inter_bezcoeffs)
end

@doc """
    Calculate the Fermi level of a representation of the band structure.

"""
function calc_fl(epm::Union{epm₋model,epm₋model2D},ebs::bandstructure;
        window::Union{Nothing,Vector{<:Real}}=ebs.fermilevel_interval, 
        ctype="mean", fermi_area::Real=epm.fermiarea)

    if ctype == "mean"
        cfun = mean 
    elseif ctype == "min"
        cfun = minimum 
    elseif ctype == "max"
        cfun = maximum
    else 
        error("Invalid ctype.")
    end
    
    simplex_bpts = sample_simplex(2,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    
    ibz_area = epm.ibz.volume
    maxsheet = round(Int,epm.electrons/2) + 2
    
    if window == nothing || window == [0,0]
        E₁ = minimum(ebs.eigenvals[1,5:end])
        E₂ = maximum(ebs.eigenvals[maxsheet,5:end])
    else
       E₁,E₂ = window
    end
    # Make sure the window contains the approx. Fermi level.
    dE = 2(E₂ - E₁)
    f₁ = sum([quad_area₋volume([simplex_pts[tri]; [minimum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₁],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
    iters₁ = 0
    while f₁ > 0
        iters₁ += 1; E₁ -= dE; dE *= 2
        if iters₁ > def_fl_max_iters
            error("Unable to find a lower limit for the rooting finding algorithm after
            $(def_fl_max_iters) iterations.")
        end
        f₁ = sum([quad_area₋volume([simplex_pts[tri]; [minimum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₁],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
    end

    dE = 2(E₂ - E₁)
    f₂ = sum([quad_area₋volume([simplex_pts[tri]; [maximum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₂],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
    iters₂ = 0
    while f₂ < 0
        iters₂ += 1; E₂ += dE; dE *= 2
        if iters₂ > def_fl_max_iters
            error("Unable to find an upper limit for the rooting finding algorithm after
            $(def_fl_max_iters) iterations.")
        end
        f₂ = sum([quad_area₋volume([simplex_pts[tri]; [maximum(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E₂],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area
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
        f = sum([quad_area₋volume([simplex_pts[tri]; [cfun(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- E],"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area 

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
    calc_flbe!(epm,ebs)

Calculate the Fermi level and band energy for a given rep. of the band struct.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ebs::bandstructure`: the band structure container 

# Returns
- `ebs::bandstructure`: update values within container for the band energy error,
    Fermi area error, Fermi level interval, Fermi area interval, band energy
    interval, and the partially occupied sheets.
"""
function calc_flbe!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure,
    inside::Bool=def_inside)

    if inside model = epm else model = nothing end

    simplex_bpts = sample_simplex(2,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]

    fl = calc_fl(epm,ebs,fermi_area=epm.fermiarea,ctype="mean")
    fl₁ = calc_fl(epm,ebs,fermi_area=epm.fermiarea,ctype = "max")
    fl₀ = calc_fl(epm,ebs,fermi_area=epm.fermiarea,ctype = "min")

    mesh_fa₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)']
                    ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    mesh_fa₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)']
                    ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]

    fa₀,fa₁ = sum(sum(mesh_fa₀)),sum(sum(mesh_fa₁))
    be = sum([quad_area₋volume([simplex_pts[tri]; [mean(ebs.mesh_intcoeffs[tri][sheet],dims=1)...]' .- fl], "volume") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])

    mesh_fa₋errs = mesh_fa₁ .- mesh_fa₀
     
    # Determine which triangles and sheets are partially occupied.
    partial_occ = [[(
        if (isapprox(mesh_fa₁[tri][sheet],0,atol=ebs.atol) &&
            isapprox(mesh_fa₀[tri][sheet],0,atol=ebs.atol))
            2
        elseif isapprox(mesh_fa₋errs[tri][sheet],0,atol=ebs.atol)
            0
        else
            1
        end
    ) for sheet=1:epm.sheets] for tri = 1:length(ebs.simplicesᵢ)]

    sigmas = [findlast(x->x==0,partial_occ[i]) for i=1:length(partial_occ)]
    partials = [findall(x->x==1,partial_occ[i]) for i=1:length(partial_occ)]

    sigma_coeffs = [
        (if sigmas[i] == nothing
            zeros(2,6)
        else
            get_intercoeffs(i,ebs.mesh,ebs.ext_mesh,ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,sigma=sigmas[i],epm=model,
            neighbor_method = ebs.neighbor_method) 
        end) for i=1:length(ebs.simplicesᵢ)]

    sigma_be₀ = [(
        if sigmas[i] == nothing
            0
        else
            simplex_size(simplices[i])*mean(sigma_coeffs[i][1][1,:] .- sigmas[i]*fl)
        end) for i=1:length(sigma_coeffs)]
    sigma_be₁ = [(
        if sigmas[i] == nothing
            0
        else
            simplex_size(simplices[i])*mean(sigma_coeffs[i][1][2,:] .- sigmas[i]*fl)
        end) for i=1:length(sigma_coeffs)]
    
    sigma_be_errs = abs.(sigma_be₁ - sigma_be₀)
    
    partial_be₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)']
                    ,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]
    partial_be₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)']
                    ,"volume") for sheet=partials[tri]] for tri=1:length(ebs.simplicesᵢ)]

    part_be_errs = [(
        if partial_be₁[i] == []
            0
        else
            sum(abs.(partial_be₁[i] - partial_be₀[i]))
        end) for i=1:length(partial_be₀)]
    
    simplices_be₋errs = abs.(sigma_be_errs + part_be_errs)
    
    spg = length(epm.pointgroup)
    be₀ = 2*spg*(sum(sigma_be₀) + sum(reduce(vcat,partial_be₀)) + fl₁*epm.fermiarea)
    be₁ = 2*spg*(sum(sigma_be₁) + sum(reduce(vcat,partial_be₁)) + fl₀*epm.fermiarea)
    
    spg = length(epm.pointgroup)
    ebs.bandenergy_errors = 2*spg.*abs.(simplices_be₋errs)
    ebs.fermiarea_errors = spg.*sum(mesh_fa₋errs)
    ebs.fermilevel_interval = [fl₀,fl₁]
    ebs.fermiarea_interval = spg.*[fa₀,fa₁]
    ebs.bandenergy_interval = [be₀,be₁]
    ebs.partially_occupied = partial_occ
    ebs.bandenergy = 2*spg*(be + fl*epm.fermiarea)
    ebs.fermilevel = fl
    ebs
end

@doc """
    refine_mesh!(epm,ebs)

Perform one iteration of adaptive refinement. See the composite type 
`bandstructure` for refinement options. 
"""
function refine_mesh!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure,
    inside::Bool=def_inside)
       
    if inside model = epm else model = nothing end 
     
    spatial = pyimport("scipy.spatial")
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    err_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.target_accuracy

    n = def_min_split_triangles
    # Refine the tile with the most error
    if ebs.refine_method == 1
        splitpos = sortperm(ebs.bandenergy_errors,rev=true)
        if length(splitpos) > n splitpos = splitpos[1:n] end

    # Refine the tiles with too much error (given the tiles' sizes).
    elseif ebs.refine_method == 2
        splitpos = filter(x -> x>0,[ebs.bandenergy_errors[i] > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])

    # Refine a fraction of the number of tiles that have too much error.
    elseif ebs.refine_method == 3
        splitpos = filter(x -> x>0,[ebs.bandenergy_errors[i] > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])
        if length(splitpos) > n
            order = sortperm(ebs.bandenergy_errors[splitpos],rev=true)
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        end            
    else
        ArgumentError("The refinement method has to be and integer equal to 1, 2 or 3.")
    end
    frac_split = length(splitpos)/length(ebs.bandenergy_errors)
    if splitpos == []
        return ebs
    end

    # A single point at the center of the triangle
    if ebs.sample_method == 1
        new_meshpts = reduce(hcat,[barytocart([1/3,1/3,1/3],s) for s=simplices[splitpos]])
    # Point at the midpoints of all edges of the triangle
    elseif ebs.sample_method == 2
        new_meshpts = reduce(hcat,[barytocart([0 1/2 1/2; 1/2 0 1/2; 1/2 1/2 0],s) for s=simplices[splitpos]])
    # If the error is 2x greater than the tolerance, split edges. Otherwise,
    # sample at the center of the triangle.
    elseif ebs.sample_method == 3
        sample_type = [
            ebs.bandenergy_errors[i] > def_allowed_err_ratio*err_cutoff[i] ? 2 : 1 for i=splitpos]
        new_meshpts = reduce(hcat,[sample_type[i] == 1 ? 
        barytocart([1/3,1/3,1/3],simplices[splitpos[i]]) :
        barytocart([0 1/2 1/2; 1/2 0 1/2; 1/2 1/2 0],simplices[splitpos[i]])
        for i=1:length(splitpos)])
    else
        ArgumentError("The sample method for refinement has to be an integer with a value of 1 or 2.")
    end

    # Remove duplicates from the new mesh points.
    new_meshpts = unique_points(new_meshpts,rtol=ebs.rtol,atol=ebs.atol)
    new_eigvals = eval_epm(new_meshpts,epm,rtol=ebs.rtol,atol=ebs.atol)
    ebs.eigenvals = [ebs.eigenvals new_eigvals]

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

    # The Line segments that bound the IBZ.
    ibz_linesegs = [Matrix(epm.ibz.points[i,:]') for i=epm.ibz.simplices]

    # Translations that need to be considered when calculating points outside the IBZ.
    # Assumes the reciprocal latice vectors are Minkowski reduced.
    bztrans = [[[i,j] for i=-1:1,j=-1:1]...]
    
    # The number of points in the mesh before adding new points.
    s = size(ebs.mesh.points,1)
    m = maximum(ebs.sym₋unique)

    # Indices of the new mesh points.
    new_ind = (m+1):(m+size(new_meshpts,2))

    # Indices of sym. equiv. points on and nearby the boundary of the IBZ. Pointer to the symmetrically unique point.
    sym_ind = zeros(Int,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))
  
    # Keep track of points on the IBZ boundaries.
    nₘ = 0
    # Add points to the mesh on the boundary of the IBZ.
    neighbors = zeros(Float64,2,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))

    for i=1:length(new_ind),op=epm.pointgroup,trans=bztrans
        pt = op*new_meshpts[:,i] + epm.recip_latvecs*trans
      
        if (any([isapprox(lineseg₋pt_dist(line_seg,pt,false),0,atol=ebs.atol) for line_seg=ibz_linesegs]) &&
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                        [ebs.mesh.points' new_meshpts neighbors[:,1:nₘ]],dims=1)))
            nₘ += 1
            sym_ind[nₘ] = new_ind[i]
            neighbors[:,nₘ] = pt
        end
    end

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

        if any([lineseg₋pt_dist(line_seg,pt,false) < bound_limit for line_seg=ibz_linesegs]) &&
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                    [ebs.ext_mesh.points' new_meshpts neighbors[:,1:nₑ]],dims=1))
            nₑ += 1
            sym_ind[nₑ] = new_ind[i]
            neighbors[:,nₑ] = pt
        end
    end

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
    ebs.mesh_intcoeffs = [get_intercoeffs(index,ebs.mesh,ebs.ext_mesh,
    ebs.sym₋unique,ebs.eigenvals,ebs.simplicesᵢ,ebs.fatten,ebs.num_near_neigh,
    neighbor_method=ebs.neighbor_method,epm=model) for index=1:length(ebs.simplicesᵢ)]    
    ebs
end

@doc """
    quadratic_method!(epm,ebs,init_msize,num_near_neigh,fermiarea_eps,target_accuracy,
        fermilevel_method,refine_method,sample_method,rtol,atol,uniform)

Calculate the band energy using uniform or adaptive quadratic integation.
"""
function quadratic_method!(epm::Union{epm₋model2D,epm₋model};
    init_msize::Int=def_init_msize, num_near_neigh::Int=def_num_near_neigh,
    fermiarea_eps::Real=def_fermiarea_eps,target_accuracy::Real=def_target_accuracy,
    fermilevel_method::Int=def_fermilevel_method, 
    refine_method::Int=def_refine_method,
    sample_method::Int=def_sample_method, 
    neighbor_method::Int=def_neighbor_method, 
    fatten::Real=def_fatten,
    rtol::Real=def_rtol,
    atol::Real=def_atol,
    uniform::Bool=def_uniform,
    inside::Bool=def_inside)
    
    ebs = init_bandstructure(epm,init_msize=init_msize, num_near_neigh=num_near_neigh,
        fermiarea_eps=fermiarea_eps, target_accuracy=target_accuracy, 
        fermilevel_method=fermilevel_method, refine_method=refine_method,
        sample_method=sample_method, neighbor_method=neighbor_method,
        fatten=fatten, inside=inside, rtol=rtol, atol=atol)
    calc_flbe!(epm,ebs,inside)

    if uniform return ebs end
    
    counter = 0
    while sum(ebs.bandenergy_errors) > ebs.target_accuracy
        counter += 1; refine_mesh!(epm,ebs); calc_flbe!(epm,ebs);
        if counter > max_refine_steps
            @warn "Failed to calculate the band energy to within the desired accuracy $(ebs.target_accuracy) after 100 iterations."
            break
        end
        println("Approx. error: ", sum(ebs.bandenergy_errors))
    end
    ebs
end

end # module