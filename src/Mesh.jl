module Mesh

# include("Geometry.jl")
# include("Polynomials.jl")

using ..Geometry: simplex_size, barytocart,lineseg₋pt_dist
using ..Polynomials: sample_simplex
using SymmetryReduceBZ.Utilities: unique_points

using PyCall: pyimport,PyObject
using QHull: Chull
using Statistics: mean
using LinearAlgebra: norm

@doc """
    get₋neighbors(index,mesh,num₋neighbors=2)

Calculate the nth-nearest neighbors of a point in a mesh.

# Arguments
- `index::Int`: the index of the point in the mesh. The coordinates
    of the point are `mesh.points[index,:]`.
- `mesh::PyObject`: a Delaunay triangulation of the mesh from `Delaunay.delaunay`.
- `num₋neighbors::Int=2`: the number of neighbors to find. For example,
    if 2, find first and second nearest neighbors.

# Returns
- `indices::AbstractVector{Int}`: the indices of neighboring points.

# Examples
```jldoctest
import Pebsi.Mesh: get₋neighbors
import PyCall: pyimport
spatial = pyimport("scipy.spatial")
pts = [0.0 0.0; 0.25 0.0; 0.5 0.0; 0.25 0.25; 0.5 0.25; 0.5 0.5]
index = 2
mesh = spatial.Delaunay(pts)
get₋neighbors(index,mesh)
# output
5-element Array{Int64,1}:
 4
 1
 5
 3
```
"""
function get₋neighbors(index::Int,mesh::PyObject,
    num₋neighbors::Int=2)::AbstractVector{Int}
    indices,indptr = mesh.vertex_neighbor_vertices
    indices .+= 1
    indptr .+= 1
    neighborsᵢ = Vector{Int64}(indptr[indices[index]:indices[index+1]-1])
    # The mesh is enclosed in a box. Don't include neighbors that are the points
    # of the box.
    neighborsᵢ = filter(x->!(x in [1,2,3,4,index]), unique(neighborsᵢ))
    for _=2:num₋neighbors
         first₋neighborsᵢ = reduce(vcat,[indptr[indices[k]:indices[k+1]-1] for k=neighborsᵢ])
         first₋neighborsᵢ = filter(x->!(x in [1,2,3,4,index]), unique(first₋neighborsᵢ))
         neighborsᵢ = [neighborsᵢ;first₋neighborsᵢ]
     end
     unique(neighborsᵢ)
end

@doc """
    ibz_init₋mesh(ibz,n;rtol,atol)

Create a triangulation of a roughly uniform mesh over the IBZ.

# Arguments
- `ibz::Chull{<:Real}`: the irreducible Brillouin zone as a convex hull object.
- `n::Int`: a measure of the number of points. The number of points over the IBZ
    will be approximately `n^2/2`.
- `rtol::Real=sqrt(eps(maximum(ibz.points)))`: a relative tolerance for finite
    precision comparisons.
- `atol::Real=1e-9`: an absolute tolerance for finite precision comparisons.

# Returns
- `mesh::PyObject`: a triangulation of a uniform mesh over the IBZ. To avoid
    collinear triangles at the boundary of the IBZ, the IBZ is enclosed in a 
    square. The first four points are the corners of the square and need to be 
    disregarded in subsequent computations.

# Examples
```jldoctest
import Pebsi.EPMs: m2ibz
import Pebsi.Mesh: ibz_init₋mesh
n = 5
ibz_init₋mesh(ibz,n)
# output
PyObject <scipy.spatial.qhull.Delaunay object at 0x19483d130>
```
"""
function ibz_init₋mesh(ibz::Chull{<:Real},n::Int;
    rtol::Real=sqrt(eps(maximum(ibz.points))),atol::Real=1e-9)::PyObject
    spatial = pyimport("scipy.spatial")

    dim = size(ibz.points,2)
    # We need to enclose the IBZ in a box to prevent collinear triangles.
    box_length = 2*maximum(abs.(ibz.points))
    if dim == 2
        box_pts = reduce(hcat,[[mean(ibz.points,dims=1)...] + box_length*[i,j] 
            for i=[-1,1] for j=[-1,1]])
    elseif dim == 3
        box_pts = reduce(hcat,[[mean(ibz.points,dims=1)...] + box_length*[i,j,k] 
            for i=[-1,1] for j=[-1,1] for k=[-1,1]])  
    else
        ArgumentError("Dimension of IBZ must be 2 or 3.")
    end
        
    mesh = spatial.Delaunay(ibz.points)
    simplices = [Array(mesh.points[mesh.simplices[i,:].+1,:]') 
        for i=1:size(mesh.simplices,1)]
    
    pt_sizes = [sqrt(simplex_size(s)/ibz.volume*n^2/2) for s=simplices]

    # Make the number of points integer multiples of the smallest point size.
    m = minimum(pt_sizes)
    mr = round(Int,m)
    if mr == 0 mr += 1 end
    pt_sizes = mr*round.(Int,pt_sizes./m)
    pts = unique_points(reduce(hcat,[barytocart(sample_simplex(
        dim,pt_sizes[i]),simplices[i]) for i=1:length(pt_sizes)]),atol=atol,rtol=rtol)
    mesh = spatial.Delaunay([box_pts'; pts'])
end

@doc """
    get_sym₋unique(mesh,pointgroup;rtol,atol)

Calculate the symmetrically unique points within the IBZ.

# Arguments
- `mesh::PyObject`: a triangulation of a mesh over the IBZ.
- `pointgroup::Vector{Matrix{Float64}}`: the point operators of the real-space
    lattice. They operate on points in Cartesian coordinates.
- `rtol::Real=sqrt(eps(maximum(real_latvecs)))`: a relative tolerance.
- `atol::Real=1e-9`: an absolute tolerance.

# Returns
- `sym₋unique::AbstractVector{<:Int}`: a vector that gives the position of the k-point
    that is equivalent to each k-point (except for the first 4 points or the
    points of the box).

# Examples
```jldoctest
import Pebsi.EPMs: m2ibz,m2pointgroup
import Pebsi.Mesh: ibz_init₋mesh
n = 5
mesh = ibz_init₋mesh(ibz,n)
get_sym₋unique(mesh,m2pointgroup)
# output
19-element Vector{Int64}:
  0
  0
  0
  0
  5
  6
  7
  8
  9
 10
 11
 12
 13
 14
 15
 16
 17
 18
 19
```
"""
function get_sym₋unique(mesh::PyObject,pointgroup::Vector{Matrix{Float64}};
    rtol::Real=sqrt(eps(maximum(mesh.points))),atol::Real=1e-9)::AbstractVector{<:Int}

    # Calculate the unique points of the uniform IBZ mesh.
    sym₋unique = zeros(Int,size(mesh.points,1))
    for i=5:size(mesh.points,1)
        # If this point hasn't been added already, add it to the list of unique points.
        if sym₋unique[i] == 0
            sym₋unique[i] = i
        end

        for pg=pointgroup
            test = [mapslices(x->isapprox(x,pg*mesh.points[i,:],atol=atol,
                rtol=rtol),mesh.points,dims=2)...]
            pos = findall(x->x==1,test)
            if pos == []
                continue
            elseif sym₋unique[pos[1]] == 0
                sym₋unique[pos[1]] = i
            end
        end
    end
    sym₋unique
end

@doc """
    notbox_simplices(mesh)

Determine all simplices in a triangulation that do not contain a box point.

# Arguments
- `mesh::PyObject`: a triangulation of a mesh over the IBZ enclosed in a box. It
    is assumed that the first four points in the mesh are the box points.

# Returns
    `simplicesᵢ::Vector{Vector{Int}}`: the simplices of the triangulation without
        box points.

# Examples
```jldoctest
using PyCall
spatial = pyimport("scipy.spatial")
pts = [-0.4940169358562923 -0.9141379262169073; -0.4940169358562923 0.24056261216234398; 0.6606836025229589 -0.9141379262169073; 0.6606836025229589 0.24056261216234398; 0.0 -0.5773502691896256; 0.06249999999999997 -0.541265877365274; 0.12499999999999994 -0.5051814855409225; 0.18749999999999992 -0.4690970937165708; 0.2499999999999999 -0.4330127018922192; 0.0 -0.4330127018922192; 0.06249999999999997 -0.3969283100678676; 0.12499999999999994 -0.360843918243516; 0.18749999999999992 -0.3247595264191644; 0.0 -0.2886751345948128; 0.06249999999999997 -0.25259074277046123; 0.12499999999999994 -0.2165063509461096; 0.0 -0.1443375672974064; 0.06249999999999997 -0.1082531754730548; 0.0 0.0]
mesh = spatial.Delaunay(pts)
notbox_simplices(mesh)
# output
16-element Vector{Vector{Int64}}:
 [19, 17, 18]
 [15, 17, 14]
 [6, 10, 5]
 [10, 11, 14]
 [17, 15, 18]
 [15, 16, 18]
 [13, 15, 12]
 [15, 13, 16]
 [8, 13, 12]
 [13, 8, 9]
 [11, 8, 12]
 [8, 11, 7]
 [6, 11, 10]
 [11, 6, 7]
 [15, 11, 12]
 [11, 15, 14]
```
"""
function notbox_simplices(mesh::PyObject)::Vector{Vector{Int}}
    simplicesᵢ = Vector{Any}(zeros(size(mesh.simplices,1)))
    n = 0
    for i=1:size(mesh.simplices,1)
        if !any([j in (mesh.simplices[i,:] .+ 1) for j=1:4])
            n += 1
            simplicesᵢ[n] = mesh.simplices[i,:] .+ 1
        end
    end
    Vector{Vector{Int}}(simplicesᵢ[1:n])    
end

@doc """
    get_cvpts(mesh,ibz,atol=1e-9)

Determine which points are on the boundary of the IBZ (or any convex hull).

# Arguments
- `mesh::PyObject`: a triangulation of a mesh over the IBZ.
- `ibz::Chull`: the irreducible Brillouin zone as a convex hull object.
- `atol::Real=1e-9`: an absolute tolerance for comparing distances to zero.

# Returns
- `cv_pointsᵢ::AbstractVector{<:Int}`: the indices of points that lie on the boundary
    of the IBZ (or convex hull).

# Examples
```jldoctest
import Pebsi.EPMs: m2ibz
using PyCall
spatial = pyimport("scipy.spatial")
import Pebsi.QuadraticIntegration: get_cvpts
pts = [-0.4940169358562923 -0.9141379262169073; -0.4940169358562923 0.24056261216234398; 0.6606836025229589 -0.9141379262169073; 0.6606836025229589 0.24056261216234398; 0.0 -0.5773502691896256; 0.0357142857142857 -0.5567306167185675; 0.0714285714285714 -0.5361109642475095; 0.1071428571428571 -0.5154913117764514; 0.1428571428571428 -0.49487165930539334; 0.1785714285714285 -0.4742520068343353; 0.2142857142857142 -0.4536323543632772; 0.2499999999999999 -0.4330127018922192; 0.0 -0.49487165930539334; 0.0357142857142857 -0.4742520068343353; 0.0714285714285714 -0.4536323543632772; 0.1071428571428571 -0.4330127018922192; 0.1428571428571428 -0.4123930494211611; 0.1785714285714285 -0.3917733969501031; 0.2142857142857142 -0.371153744479045; 0.0 -0.4123930494211612; 0.0357142857142857 -0.39177339695010305; 0.0714285714285714 -0.37115374447904503; 0.1071428571428571 -0.3505340920079869; 0.1428571428571428 -0.3299144395369289; 0.1785714285714285 -0.3092947870658709; 0.0 -0.3299144395369289; 0.0357142857142857 -0.3092947870658708; 0.0714285714285714 -0.28867513459481275; 0.1071428571428571 -0.26805548212375474; 0.1428571428571428 -0.24743582965269667; 0.0 -0.24743582965269667; 0.0357142857142857 -0.2268161771816386; 0.0714285714285714 -0.20619652471058056; 0.1071428571428571 -0.1855768722395225; 0.0 -0.16495721976846445; 0.0357142857142857 -0.14433756729740638; 0.0714285714285714 -0.12371791482634834; 0.0 -0.08247860988423222; 0.0357142857142857 -0.06185895741317417; 0.0 0.0]
mesh = spatial.Delaunay(pts)
get_cvpts(mesh,m2ibz)
# output
21-element Vector{Int64}:
  5
  6
  7
  8
  9
 10
 11
 12
 13
 19
 20
 25
 26
 30
 31
 34
 35
 37
 38
 39
 40
```
"""
function get_cvpts(mesh::PyObject,ibz::Chull;atol::Real=1e-9)::AbstractVector{<:Int}
    
    ibz_linesegs = [Matrix(ibz.points[i,:]') for i=ibz.simplices]
    cv_pointsᵢ = [0 for i=1:size(mesh.points,1)]
    n = 0
    for i=1:size(mesh.points,1)
        if any([isapprox(lineseg₋pt_dist(line_seg,mesh.points[i,:]),0,atol=atol) 
            for line_seg=ibz_linesegs])
            n += 1
            cv_pointsᵢ[n] = i
        end
    end

    cv_pointsᵢ[1:n]
end

@doc """
    get_extmesh(ibz,mesh,pointgroup,recip_latvecs,near_neigh=1;rtol,atol)

Calculate a triangulation of points within and just outside the IBZ.

# Arguments
- `ibz::Chull`: the irreducible Brillouin zone as a convex hull object.
- `mesh::PyObject`: a triangulation of a mesh over the IBZ.
- `pointgroup::Vector{Matrix{Float64}}`: the point operators of the real-space
    lattice.
- `recip_latvecs::AbstractMatrix{<:Real}`: the reciprocal lattice vectors as 
    columns of a matrix.
- `near_neigh::Int=1`: the number of nearest neighbors to include outside the 
    IBZ.
- `rtol::Real=sqrt(eps(maximum(abs.(mesh.points))))`: a relative tolerance for 
    floating point comparisons.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `::PyObject`: a triangulation of points within and without the IBZ. The points
    outside the IBZ are rotationally or translationally equivalent to point inside
    the IBZ.
- `sym₋unique::AbstractVector{<:Int}`: a vector that gives the position of the k-point
    that is equivalent to each k-point (except for the first 4 points or the
    points of the box).

# Examples
```jldoctest
using PyCall
spatial = pyimport("scipy.spatial")
import Pebsi.EPMs: m2ibz,m2pointgroup,m2recip_latvecs
pts = [-0.4940169358562923 -0.9141379262169073; -0.4940169358562923 0.24056261216234398; 0.6606836025229589 -0.9141379262169073; 0.6606836025229589 0.24056261216234398; 0.0 -0.5773502691896256; 0.0357142857142857 -0.5567306167185675; 0.0714285714285714 -0.5361109642475095; 0.1071428571428571 -0.5154913117764514; 0.1428571428571428 -0.49487165930539334; 0.1785714285714285 -0.4742520068343353; 0.2142857142857142 -0.4536323543632772; 0.2499999999999999 -0.4330127018922192; 0.0 -0.49487165930539334; 0.0357142857142857 -0.4742520068343353; 0.0714285714285714 -0.4536323543632772; 0.1071428571428571 -0.4330127018922192; 0.1428571428571428 -0.4123930494211611; 0.1785714285714285 -0.3917733969501031; 0.2142857142857142 -0.371153744479045; 0.0 -0.4123930494211612; 0.0357142857142857 -0.39177339695010305; 0.0714285714285714 -0.37115374447904503; 0.1071428571428571 -0.3505340920079869; 0.1428571428571428 -0.3299144395369289; 0.1785714285714285 -0.3092947870658709; 0.0 -0.3299144395369289; 0.0357142857142857 -0.3092947870658708; 0.0714285714285714 -0.28867513459481275; 0.1071428571428571 -0.26805548212375474; 0.1428571428571428 -0.24743582965269667; 0.0 -0.24743582965269667; 0.0357142857142857 -0.2268161771816386; 0.0714285714285714 -0.20619652471058056; 0.1071428571428571 -0.1855768722395225; 0.0 -0.16495721976846445; 0.0357142857142857 -0.14433756729740638; 0.0714285714285714 -0.12371791482634834; 0.0 -0.08247860988423222; 0.0357142857142857 -0.06185895741317417; 0.0 0.0]
mesh = spatial.Delaunay(pts)
get_extmesh(m2ibz,mesh,m2pointgroup,m2recip_latvecs)
# output
PyObject (<scipy.spatial.qhull.Delaunay object at 0x1802f7820>, array([ 0,  0,  0,  0,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
       18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
       35, 36, 37, 38, 39, 40, 13, 13,  6,  6,  7,  7, 15, 15, 14, 14, 16,
       10, 17, 17, 11, 18, 18, 18, 19, 24, 21, 27, 22, 29, 33, 33, 35, 32,
       32, 37, 36, 36, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39],
      dtype=int64))
```
"""
function get_extmesh(ibz::Chull,mesh::PyObject,pointgroup::Vector{Matrix{Float64}},
    recip_latvecs::AbstractMatrix{<:Real},near_neigh::Int=1;
    rtol::Real=sqrt(eps(maximum(abs.(mesh.points)))),atol::Real=1e-9)

    spatial = pyimport("scipy.spatial")
    sym₋unique = get_sym₋unique(mesh,pointgroup);
    cv_pointsᵢ = get_cvpts(mesh,ibz)
    neighborsᵢ = reduce(vcat,[get₋neighbors(i,mesh,near_neigh) for i=cv_pointsᵢ]) |> unique
    
    numpts = size(mesh.points,1)
    # Calculate the maximum distance between neighboring points
    bound_limit = 1.01*maximum(reduce(vcat,[[norm(mesh.points[i,:] - mesh.points[j,:]) 
                    for j=get₋neighbors(i,mesh,near_neigh)] for i=cv_pointsᵢ]))

    ibz_linesegs = [Matrix(ibz.points[i,:]') for i=ibz.simplices]
    bztrans = [[[i,j] for i=-1:1,j=-1:1]...]

    # Rotate the neighbors of the points on the boundary. Keep the points if they are within
    # a distance of `bound_limit` of any of the interior boundaries.
    neighbors = zeros(Float64,2,length(neighborsᵢ)*length(pointgroup)*length(bztrans))
    sym₋unique = [sym₋unique; zeros(Int,size(neighbors,2))]
    n = 0
    for i=neighborsᵢ,op=pointgroup,trans=bztrans
        pt = op*mesh.points[i,:] + recip_latvecs*trans
        if any([lineseg₋pt_dist(line_seg,pt,false) < bound_limit for line_seg=ibz_linesegs]) &&
            !any(mapslices(x->isapprox(x,pt,atol=atol,rtol=rtol),[mesh.points' neighbors[:,1:n]],dims=1))
            n += 1
            neighbors[:,n] = pt
            sym₋unique[numpts + n] = sym₋unique[i]
        end
    end
    neighbors = neighbors[:,1:n]
    sym₋unique = sym₋unique[1:numpts + n]
    (spatial.Delaunay(unique_points([mesh.points; neighbors']',
        rtol=rtol,atol=atol)'),sym₋unique)
end

end # Module