module QuadraticIntegration

using SymmetryReduceBZ.Utilities: unique_points, shoelace
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ..Polynomials: eval_poly,getpoly_coeffs,getbez_pts₋wts,eval_bezcurve,
    conicsection
using ..EPMs: eval_epm, RytoeV, epm₋model, epm₋model2D
using ..Mesh: get₋neighbors,notbox_simplices,get_cvpts,ibz_init₋mesh, 
    get_extmesh
using ..Geometry: order_vertices!,simplex_size,insimplex,barytocart,carttobary,
    sample_simplex,lineseg₋pt_dist

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
- `num_neigh::Int`: the number of nearest neighbors to consider to include when
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
    a single point at the center of the triangle. 2-add point the midpoints of 
    all three edges.
- `rtol::Real`: a relative tolerance for floating point comparisons.
- `atol::Real`: an absolute tolerance for floating point comparisons.
- `mesh::PyObject`: a Delaunay triangulation of points over the IBZ.
- `simplicesᵢ::Vector{Vector{Int}}`: the indices of points at the corners of the 
    tile for all tiles in the triangulation.
- `ext_mesh::PyObject`: a Delaunay triangulation of points within and around the 
    IBZ. The number of points outside is determined by `num_neigh`.
- `sym₋unique::AbstractVector{<:Int}`: the indices of symmetrically unique points
    in the mesh.
- `eigenvals::AbstractMatrix{<:Real}`: the eigenvalues at each of the points unique
    by symmetry.
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
    num_neigh::Int
    fermiarea_eps::Real
    target_accuracy::Real
    fermilevel_method::Int
    refine_method::Int
    sample_method::Int
    rtol::Real
    atol::Real
    mesh::PyObject
    simplicesᵢ::Vector{Vector{Int}}
    ext_mesh::PyObject
    sym₋unique::AbstractVector{<:Int}
    eigenvals::AbstractMatrix{<:Real}
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
    init_bandstructure(epm; init_msize,nut_neigh,fermiarea_eps,target_accuracy,
        fermilevel_method,refine_method,sample_method,rtol,atol)

Initialize a band structure container

# Arguments
- `epm::Union{epm₋model,epm₋model2D}`: an empirical pseudopotential. 
- `init_msize::Int`: the initial size of the mesh over the IBZ. The number of 
    points is approximately proportional to init_msize^2/2.
- `num_neigh::Int`: the number of nearest neighbors to consider to include when
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
    a single point at the center of the triangle. 2-add point the midpoints of 
    all three edges.
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
    init_msize::Int=5,
    num_neigh::Int=1,
    fermiarea_eps::Real=1e-6,
    target_accuracy::Real=1e-4,
    fermilevel_method::Int=1,
    refine_method::Int=1,
    sample_method::Int=1,
    num_cores::Integer=1,
    rtol::Real=1e-9,
    atol::Real=1e-9)

    mesh = ibz_init₋mesh(epm.ibz,init_msize;rtol=rtol,atol=atol)
    simplicesᵢ = notbox_simplices(mesh)
    ext_mesh,sym₋unique = get_extmesh(epm.ibz,mesh,epm.pointgroup,epm.recip_latvecs,num_neigh;
        rtol=rtol,atol=atol)
      
    uniqueᵢ = sort(unique(sym₋unique))[2:end]
     
    if num_cores == 1    
        eigenvals = eval_epm(mesh.points[uniqueᵢ,:]', epm, rtol=rtol, atol=atol) 
    else
        eigenvals = reduce(hcat,pmap(x->eval_epm(x,epm,rtol=rtol,atol=atol),
            [mesh.points[i,:] for i=1:size(mesh.points,1)]))        
    end

    if typeof(epm) == epm₋model2D
        eigenvals = [zeros(epm.sheets,4) eigenvals]
    else
        eigenvals = [zeros(epm.sheets,8) eigenvals]
    end        

    # eigenvals = zeros(epm.sheets,length(uniqueᵢ)+4)
    # for i=uniqueᵢ
    #     eigenvals[:,i] = eval_epm(mesh.points[i,:], epm, rtol=rtol, atol=atol)
    # end
    
    mesh_intcoeffs = [get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,
        simplicesᵢ) for index=1:length(simplicesᵢ)];
    
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
        num_neigh,
        fermiarea_eps,
        target_accuracy,
        fermilevel_method,
        refine_method,
        sample_method,
        rtol,
        atol,
        
        mesh,
        simplicesᵢ,
        ext_mesh,
        sym₋unique,
        eigenvals,
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
function simpson(interval_len::Real,vals::AbstractVector{<:Real})
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
function quadval_vertex(bezcoeffs::AbstractVector{<:Real})
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
end

@doc """
    edge_intersects(bezpts;atol)

Calculate where a quadratic curve is equal to zero within [0,1).

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points (columns of an array).
- `atol::Real=1e-9`: absolute tolerance for comparisons of floating point 
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
    atol::Real=1e-9)::AbstractVector{<:Real}
   
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
    atol::Real=1e-9)::Array
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
    atol::Real=1e-9)::AbstractVector{<:Real}
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
    allpts::AbstractArray=[]; atol::Real=1e-9)::AbstractArray
    spatial = pyimport("scipy.spatial")
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
            allpts = pts
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
    # Had to add box points to prevent collinear triangles.
    xmax,ymax = maximum(bezpts[1:2,:],dims=2)
    xmin,ymin = minimum(bezpts[1:2,:],dims=2)
    xmax += 0.2*abs(xmax - xmin)
    xmin -= 0.2*abs(xmax - xmin)
    ymax += 0.2*abs(ymax - ymin)
    ymin -= 0.2*abs(ymax - ymin)
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
function split_bezsurf(bezpts::AbstractMatrix{<:Real};atol=1e-9)::AbstractArray
    
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
    analytic_coeffs(coeffs,w;atol=1e-9)

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
function analytic_volume(coeffs::AbstractVector{<:Real},w::Real;
        atol::Real=1e-9)::Real
    
    (c₅,c₃,c₀,c₄,c₁,c₂) = coeffs
    d = c₀+c₁+c₂+c₃+c₄+c₅
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
    quantity::String; atol::Real=1e-9)::Real

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
    quad_area₋volume(bezpts,quantity;atol=1e-9)

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
        quantity::String;atol::Real=1e-9)::Real
    sum([two₋intersects_area₋volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])    
end

@doc """
    calc_mesh₋bezcoeffs(epm,mesh;rtol,atol)

Calculate the quadratic coefficients of a triangulation of the IBZ.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential structure.
- `mesh::PyObject`: a simplex tesselation of the IBZ.
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
function calc_mesh₋bezcoeffs(epm::Union{epm₋model2D,epm₋model} ,mesh::PyObject;
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))),
    atol::Real=1e-9)::Vector{Vector{Any}}

    deg=2
    dim = size(epm.real_latvecs,1)
    simplex_bpts=sample_simplex(dim,deg)
    mesh_bezcoeffs = [Vector{Any}[] for i=1:size(mesh.simplices,1)]
    for s = 1:size(mesh.simplices,1)
        simplex = Array(mesh.points[mesh.simplices[s,:] .+ 1,:]')
        simplex_pts = barytocart(simplex_bpts,simplex)
        values = eval_epm(simplex_pts,epm,rtol=rtol,atol=atol)
        mesh_bezcoeffs[s] = [getpoly_coeffs(values[i,:],simplex_bpts,dim,deg) for i=1:epm.sheets]
    end
    mesh_bezcoeffs
end

@doc """
    get_intercoeffs(index,mesh,ext_mesh,sym₋unique,eigenvals,simplicesᵢ)

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

num_neigh = 2
ext_mesh,sym₋unique = get_extmesh(m2ibz,mesh,m2pointgroup,m2recip_latvecs,num_neigh)

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
        simplicesᵢ::Vector{Vector{Int64}})::Vector{Matrix{Float64}}

    simplexᵢ = simplicesᵢ[index]
    simplex = Matrix(mesh.points[simplexᵢ,:]')
    neighborsᵢ = reduce(vcat,[get₋neighbors(s,ext_mesh,2) for s=simplexᵢ]) |> unique
    neighborsᵢ = filter(x -> !(x in simplexᵢ),neighborsᵢ)
     
    b = carttobary(ext_mesh.points[neighborsᵢ,:]',simplex)
    # b = reduce(hcat,[carttobary(ext_mesh.points[i,:],simplex) for i=neighborsᵢ])
    # M = 2*Matrix(reduce(hcat,[[b[1,i]*b[2,i], b[2,i]*b[3,i], b[3,i]*b[1,i]] for i=1:size(b,2)])')
    M = mapslices(x -> 2*[x[1]*x[2],x[1]*x[3],x[2]*x[3]],b,dims=1)'
    # Zm = Matrix(b').^2
    # Dᵢ = [sum((M[i,:]/2).^2) for i=1:size(M,1)]
    Dᵢ = mapslices(x -> sum((x/2).^2),M,dims=2)
        
    # Minimum distance from the edges of the triangle.
    # W = diagm([minimum([lineseg₋pt_dist(simplex[:,s],ext_mesh.points[i,:]) for s=[[1,2],[2,3],[3,1]]])
    #     for i=neighborsᵢ])
 
    # Distance from the center of the triangle.
    # W = diagm([norm(ext_mesh.points[i,:] - mean(simplex,dims=2)) for i=neighborsᵢ])
    
    W=I
    
    inter_bezcoeffs = [zeros(2,size(eigenvals,1)) for i=1:size(eigenvals,1)]
    for sheet = 1:size(eigenvals,1)
        fᵢ = eigenvals[sheet,sym₋unique[neighborsᵢ]]
        q = eigenvals[sheet,sym₋unique[simplexᵢ]]
        Z = fᵢ - (b.^2)'*q
        # Z = fᵢ - Zm*q

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
        c = [c[i] .+ ϵ[:,i] for i=1:3]

        c1,c2,c3 = c
        intercoeffs = reduce(hcat,[[q1,q1],c1,[q2,q2],c2,c3,[q3,q3]])
        inter_bezcoeffs[sheet] = intercoeffs
    end

    Vector{Matrix{Float64}}(inter_bezcoeffs)
end

@doc """
    Calculate the Fermi level of a representation of the band structure.

"""
function calc₋fl(epm::Union{epm₋model,epm₋model2D},ebs::bandstructure;
        window::Union{Nothing,Vector{<:Real}}=nothing,
        fermi_area::Real=epm.fermiarea)

    simplex_bpts = sample_simplex(2,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    
    ibz_area = epm.ibz.volume
    # max_sheet = round(Int,epm.electrons/2)
    max_sheet = epm.sheets

    if window == nothing
        E₁ = minimum(ebs.eigenvals[1,5:end])
        E₂ = maximum(ebs.eigenvals[max_sheet,5:end])
    else
        E₁,E₂ = window
    end
    E = (E₁ + E₂)/2

    f₁ = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_intcoeffs[tri][sheet][1,:]' .- E₁]
        ,"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets]) - fermi_area

    f₂ = max_sheet*ibz_area - fermi_area
    f₃ = 0
    E₃ = 0
    iters = 0
    f,fa₁,fa₂ = 1e9,1e9,1e9
    while abs(f) > ebs.fermiarea_eps
        iters += 1
        if iters > 50
            @warn "Failed to converge the Fermi area to within the provided tolerance of $(ebs.fermiarea_eps)."
            break
        end
        println("area error: ", abs((fa₁ + fa₂)/2 - fermi_area))

        fa₁ = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_intcoeffs[tri][sheet][1,:]' .- E]
                ,"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])
        fa₂ = sum([quad_area₋volume([simplex_pts[tri]; ebs.mesh_intcoeffs[tri][sheet][2,:]' .- E]
                ,"area") for tri=1:length(ebs.simplicesᵢ) for sheet=1:epm.sheets])
        f = (fa₁ + fa₂)/2 - fermi_area

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

            if t < 0
                t = 1e-9
            elseif t > 1
                t = 1 - 1e-9
            end
        else
            ArgumentError("The method for calculating the Fermi is either 1 or 2.")
        end
        E = E₁ + t*(E₂ - E₁)
    end
    
    (E,fa₁,fa₂)
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
function calc_flbe!(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure)

    maxsheet = round(Int,epm.electrons/2)
    window = [minimum(ebs.eigenvals[1:maxsheet+2,5:end]),
        maximum(ebs.eigenvals[1:maxsheet+2,5:end])]
    
    simplex_bpts = sample_simplex(2,2)
    (fl,fa₀,fa₁) = calc₋fl(epm,ebs,window=window,fermi_area=epm.fermiarea)
    (fl₁,null,null) = calc₋fl(epm,ebs,window=window,fermi_area=fa₀)
    (fl₀,null,null) = calc₋fl(epm,ebs,window=window,fermi_area=fa₁)

    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
        
    mesh_fa₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)']
                    ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    mesh_be₁ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][2,:] .- fl₀)']
                    ,"volume") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    mesh_fa₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)']
                    ,"area") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    mesh_be₀ = [[quad_area₋volume([simplex_pts[tri]; (ebs.mesh_intcoeffs[tri][sheet][1,:] .- fl₁)']
                    ,"volume") for sheet=1:epm.sheets] for tri=1:length(ebs.simplicesᵢ)]
    
    be = sum([quad_area₋volume([simplex_pts[tri]; mean(ebs.mesh_intcoeffs[tri][sheet],dims=1) .- fl]
                    ,"volume") for sheet=1:epm.sheets for tri=1:length(ebs.simplicesᵢ)])
    
    mesh_fa₋errs = mesh_fa₁ .- mesh_fa₀
    mesh_be₋errs = mesh_be₁ .- mesh_be₀
    
    # Determine which triangles and sheets are partially occupied.
    partial_occ = [[(isapprox(mesh_fa₁[tri][sheet],simplex_size(simplices[1]),atol=ebs.atol,rtol=ebs.rtol) ||
        isapprox(mesh_fa₀[tri][sheet],simplex_size(simplices[1]),atol=ebs.atol,rtol=ebs.rtol) ||
        isapprox(mesh_fa₁[tri][sheet],0,atol=ebs.atol) ||
        isapprox(mesh_fa₀[tri][sheet],0,atol=ebs.atol)) ? 0 : 1
        for sheet=1:epm.sheets] for tri = 1:length(ebs.simplicesᵢ)]

    # Calculate the band energy errors. Take the absolute value of errors of sheets
    # that are parially occupied and sum the errors of sheets are are occupied or unoccupied
    simplices_be₋errs = zeros(length(ebs.simplicesᵢ))
    for tri = 1:length(ebs.simplicesᵢ)
        for sheet = 1:epm.sheets
            if partial_occ[tri][sheet] == 0
                simplices_be₋errs[tri] += mesh_be₋errs[tri][sheet]
            else
                simplices_be₋errs[tri] += abs(mesh_be₋errs[tri][sheet])
            end
        end
    end
    
    spg = length(epm.pointgroup)
    ebs.bandenergy_errors = spg.*abs.(simplices_be₋errs)
    ebs.fermiarea_errors = spg.*sum(mesh_fa₋errs)
    ebs.fermilevel_interval = [fl₀,fl₁]
    ebs.fermiarea_interval = spg.*[fa₀,fa₁]
    ebs.bandenergy_interval = spg.*([sum(sum(mesh_be₀)) + fl₁*epm.fermiarea,
         sum(sum(mesh_be₁)) + fl₀*epm.fermiarea])
    ebs.partially_occupied = partial_occ
    ebs.bandenergy = spg*(be + fl*epm.fermiarea)
    ebs.fermilevel = fl
    ebs
end


@doc """
    refine_mesh(epm,ebs)

Perform one iteration of adaptive refinement. See the composite type 
`bandstructure` for refinement options. 
"""
function refine_mesh(epm::Union{epm₋model2D,epm₋model},ebs::bandstructure)
    
    spatial = pyimport("scipy.spatial")
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]

    # Refine the tile with the most error
    if ebs.refine_method == 1
        splitpos = [sortperm(ebs.bandenergy_errors)[end]]
    # Refine the tiles with too much error (given the tiles' sizes).
    elseif ebs.refine_method == 2
        err_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.target_accuracy
        splitpos = filter(x -> x>0,[ebs.bandenergy_errors[i] > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])
    else
        ArgumentError("The refinement method has to be an integer of 1 or 2.")
    end
    
    # A single point at the center of the triangle
    if ebs.sample_method == 1
        new_meshpts = reduce(hcat,[barytocart([1/3,1/3,1/3],s) for s=simplices[splitpos]])
    # Point at the midpoints of all edges of the triangle
    elseif ebs.sample_method == 2
        new_meshpts = reduce(hcat,[barytocart([0 1/2 1/2; 1/2 0 1/2; 1/2 1/2 0],s) for s=simplices[splitpos]])
    else
        ArgumentError("The sample method for refinement has to be an integer of 1 or 2.")
    end

    # Remove duplicates from the new mesh points.
    new_meshpts = unique_points(new_meshpts,rtol=ebs.rtol,atol=ebs.atol)

    cv_pointsᵢ = get_cvpts(ebs.mesh,epm.ibz,atol=ebs.atol)
    # Calculate the maximum distance between neighboring points
    bound_limit = 1.01*maximum(reduce(vcat,[[norm(ebs.mesh.points[i,:] - ebs.mesh.points[j,:]) 
                    for j=get₋neighbors(i,ebs.mesh,ebs.num_neigh)] for i=cv_pointsᵢ]))

    # The Line segments that bound the IBZ.
    ibz_linesegs = [Matrix(epm.ibz.points[i,:]') for i=epm.ibz.simplices]

    # Translations that need to be considered when calculating points outside the IBZ.
    bztrans = [[[i,j] for i=-1:1,j=-1:1]...]
    
    # Indices of the new mesh points.
    new_ind = size(ebs.mesh.points,1):size(ebs.mesh.points,1)+size(new_meshpts,2) - 1
    ebs.sym₋unique = [ebs.sym₋unique; new_ind]
    
    # Indices of sym. equiv. points on the boundary of and nearby the IBZ. Points
    # to the symmetrically unique point.
    sym_mesh = zeros(Int,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))
    
    # Keep track of points on the IBZ boundaries.
    n = 0
    # Add points to the mesh on the boundary of the IBZ.
    neighbors = zeros(Float64,2,size(new_meshpts,2)*length(epm.pointgroup)*length(bztrans))
    for i=1:length(new_ind),op=epm.pointgroup,trans=bztrans
        pt = op*new_meshpts[:,i] + epm.recip_latvecs*trans

        if (any([isapprox(lineseg₋pt_dist(line_seg,pt,false),0,atol=ebs.atol) for line_seg=ibz_linesegs]) &&
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                        [ebs.mesh.points' new_meshpts neighbors[:,1:n]],dims=1)))
            n += 1
            sym_mesh[n] = new_ind[i]
            neighbors[:,n] = pt
        end
    end
    ebs.mesh = spatial.Delaunay([ebs.mesh.points; new_meshpts'; neighbors[:,1:n]'])
    
    # Add points to the extended mesh nearby but outside of the IBZ
    for i=1:length(new_ind),op=epm.pointgroup,trans=bztrans
        pt = op*new_meshpts[:,i] + epm.recip_latvecs*trans

        if any([lineseg₋pt_dist(line_seg,pt,false) < bound_limit for line_seg=ibz_linesegs]) &&
            !any(mapslices(x->isapprox(x,pt,atol=ebs.atol,rtol=ebs.rtol),
                    [ebs.ext_mesh.points' new_meshpts neighbors[:,1:n]],dims=1))
            n += 1
            sym_mesh[n] = new_ind[i]
            neighbors[:,n] = pt
        end
    end
    
    ebs.sym₋unique = [ebs.sym₋unique; sym_mesh[1:n]]
    ebs.ext_mesh = spatial.Delaunay([ebs.ext_mesh.points; new_meshpts'; neighbors[:,1:n]'])
    new_eigvals = zeros(epm.sheets,size(new_meshpts,2))

    for i=1:size(new_meshpts,2)
        new_eigvals[:,i] = eval_epm(new_meshpts[:,i],epm,rtol=ebs.rtol,atol=ebs.atol)
    end

    ebs.simplicesᵢ = notbox_simplices(ebs.mesh)
    ebs.eigenvals = [ebs.eigenvals new_eigvals]
    ebs.mesh_intcoeffs = [get_intercoeffs(index,ebs.mesh,ebs.ext_mesh,ebs.sym₋unique,ebs.eigenvals,
        ebs.simplicesᵢ) for index=1:length(ebs.simplicesᵢ)]
    
    ebs
end
 
end # module