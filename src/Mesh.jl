module Mesh

using ..Geometry: simplex_size, barytocart, lineseg₋pt_dist, ptface_mindist,
    sample_simplex
using ..Defaults: def_atol, def_mesh_scale, def_max_neighbor_tol,
    def_neighbors_per_bin2D, def_neighbors_per_bin3D, def_num_neighbors2D, 
    def_num_neighbors3D
using SymmetryReduceBZ.Utilities: unique_points, get_uniquefacets
using PyCall: pyimport,PyObject
using QHull: Chull
using Statistics: mean
using LinearAlgebra: norm, dot

export get_neighbors, choose_neighbors, choose_neighbors3D, ibz_init₋mesh, 
    get_sym₋unique!, notbox_simplices, get_cvpts, get_extmesh, trimesh

@doc """
    get_neighbors(index,mesh,num₋neighbors=2)

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
import Pebsi.Mesh: get_neighbors
import PyCall: pyimport
spatial = pyimport("scipy.spatial")
pts = [0.0 0.0; 0.25 0.0; 0.5 0.0; 0.25 0.25; 0.5 0.25; 0.5 0.5]
index = 2
mesh = spatial.Delaunay(pts)
get_neighbors(index,mesh)
# output
2-element Vector{Int64}:
 5
 6
```
"""
function get_neighbors(index::Int,mesh::PyObject,
    num₋neighbors::Int=2)::AbstractVector{Int}
    dim = size(mesh.points,2)
    if dim == 2 ignore = collect(1:4) else ignore = collect(1:8) end
    indices,indptr = mesh.vertex_neighbor_vertices
    indices .+= 1
    indptr .+= 1
    neighborsᵢ = Vector{Int64}(indptr[indices[index]:indices[index+1]-1])
    # The mesh is enclosed in a box. Don't include neighbors that are the corners
    # of the box.
    neighborsᵢ = filter(x->!(x in [ignore; index]), unique(neighborsᵢ))
    for _=2:num₋neighbors
         first₋neighborsᵢ = reduce(vcat,[indptr[indices[k]:indices[k+1]-1] for k=neighborsᵢ])
         first₋neighborsᵢ = filter(x->!(x in [ignore;index]), unique(first₋neighborsᵢ))
         neighborsᵢ = [neighborsᵢ;first₋neighborsᵢ]
    end
    
    unique(neighborsᵢ)
end

@doc """
    choose_neighbors(simplex,neighborsᵢ,neighbors;num_neighbors)

Select neighbors to include in interval coefficient calculations.

# Arguments
- `simplex::AbstractMatrix{<:Real}`: the Cartesian coordinates of the corners of
   a triangle in columns of a matrix.
- `neighborsᵢ::AbstractVector{<:Integer}`: the positions of the neighbors in the
    extended mesh.
- `neighbors::AbstractMatrix{<:Real}`: the coordinates of the neighboring points
    in columns of a matrix.
- `num_neighbors::Integer=nothing`: the number of neighbors to include in the 
    calculation of interval coefficients.

# Returns
- `neighᵢ::AbstractVector{<:Integer}`: the positions of the neighboring points
    to include in the calculation of interval coefficients in the extended mesh.

# Examples
```jldoctest
using Pebsi.Mesh: choose_neighbors
using Pebsi.Mesh: choose_neighbors
triangle = [0. 1. 1.; 0. 0. 1.]
neighborsᵢ = [5, 6, 7, 8, 9, 10]
neighbors = [-1. -1. 0. 1. 2. 0.; 0. 1. 2. 2. 3. 4.]
num_neighbors = 4
choose_neighbors(triangle,neighborsᵢ,neighbors,num_neighbors=num_neighbors)
# output
4-element Vector{Integer}:
 5
 8
 7
 6
```
"""
function choose_neighbors(simplex::AbstractMatrix{<:Real},
    neighborsᵢ::AbstractVector{<:Integer}, neighbors::AbstractMatrix{<:Real};
    num_neighbors::Integer=nothing)::AbstractVector{<:Integer}

    dim = size(simplex,1)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end 
    neighbors_per_bin = if dim == 2 def_neighbors_per_bin2D else def_neighbors_per_bin3D end

    center = vec(mean(simplex,dims=2)) # Measure angles from the center of the triangle
    angles = [atan(neighbors[2,i]-center[2],neighbors[1,i]-center[1]) for i=1:size(neighbors,2)]
    order = sortperm(angles); neighbors = neighbors[:,order]; angles = angles[order]
    neighborsᵢ = neighborsᵢ[order]
    distances = [minimum([
        lineseg₋pt_dist(neighbors[:,j],simplex[:,[i,mod1(i+1,3)]]) for i=1:3]) for j=1:size(neighbors,2)]
    
    # Group neighboring points by angle ranges
    nbins = round(Int,num_neighbors/neighbors_per_bin)
    angle_segs = -π:2π/nbins:π;
    angle_ran = [[] for _=1:nbins] # angle ranges

    p = 1
    for (i,θ) in enumerate(angles)
        if θ <= angle_segs[p+1]
            push!(angle_ran[p],i)
        else
            p += 1
            push!(angle_ran[p],i)
        end
    end

    for p=1:nbins
        # Order the points in each bin by distance
        distances = [minimum([lineseg₋pt_dist(
            neighbors[:,j],simplex[:,[i,mod1(i+1,3)]]) for i=1:3]) for j=angle_ran[p]]
        dorder = sortperm(distances)
        angle_ran[p] = neighborsᵢ[angle_ran[p][dorder]]
    end
    
    # Select the points within the angle ranges that are closest to the sample points of the triangle.
    c = 0
    neighᵢ = Vector{Integer}(undef,0)
    while length(neighᵢ) < num_neighbors
        c += 1
        for i=1:nbins
            # Move on if there are no more points in this angle range to add.
            if length(angle_ran[i]) < c
                continue
            else
                push!(neighᵢ,angle_ran[i][c])            
            end
        end    
    end
    neighᵢ
end

face_ind = [[1,2,3],[2,3,4],[3,4,1],[4,1,2]]
@doc """
    choose_neighbors3D(simplex,neighborsᵢ,neighbors,num_neighbors)

Select neighboring points that are close and uniformly surround the tetrahedron.

# Arguments
- `simplex`: the corners of the tetrahedron as columns of a matrix in Cartesian
    coordinates.
- `neighborsᵢ`: the indices of all neighboring points.
- `neighbors`: the Cartesian coordinates of neighboring points as columns of a
    matrix.
- `num_neighbors`: the number of neighbors to keep.

# Returnts
- `neighᵢ`: the indices of neighboring points kept.

# Examples
```jldoctest
using Pebsi.Mesh: choose_neighbors3D
tetrahedron = [0. 1. 0. 0.; 0. 0. 1. 0.; 0. 0. 0. 1.]
neighborsᵢ = [9, 10, 11, 12, 13, 14, 15, 16]
neighbors = [-2. -2. -2. -2. 2. 2. 2. 2. ; -2. -2. 2. 2. -2. -2. 2. 2.; -2. 2. -2. 2. -2. 2. -2. 2.]
num_neighbors = 4
choose_neighbors3D(tetrahedron,neighborsᵢ,neighbors,num_neighbors=num_neighbors)
# output
4-element Vector{Any}:
 14
 16
 10
 15
```
"""
function choose_neighbors3D(simplex,neighborsᵢ,neighbors;num_neighbors=nothing)

    dim = size(simplex,2)-1
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end

    if length(neighborsᵢ) < num_neighbors 
        return neighborsᵢ
    end

    center = vec(mean(simplex,dims=2)) # Measure angles from the center of the triangle
    ϕs = [acos(dot(neighbors[:,i] - center,[0,0,1] - center)/(norm(neighbors[:,i] - center)*norm([0,0,1] - 
                    center))) for i=1:size(neighbors,2)]
    θs = [atan(neighbors[2,i]-center[2],neighbors[1,i]-center[1]) for i=1:size(neighbors,2)]

    orderϕ = sortperm(ϕs)
    orderθ = sortperm(θs)
    neighbors_per_bin = if dim == 2 def_neighbors_per_bin2D else def_neighbors_per_bin3D end

    nbinsθ = round(Int,√(2*num_neighbors/neighbors_per_bin))
    nbinsϕ = ceil(Int,nbinsθ/2)
    segsθ = -π:2π/nbinsθ:π
    segsϕ = 0:π/nbinsϕ:π

    binsθ = [[] for _=1:nbinsθ]
    n = 1
    for i in orderθ
        if θs[i] > segsθ[n+1] && !isapprox(θs[i],segsθ[n+1])
            n += 1
        end
        push!(binsθ[n], i)
    end

    binsϕ = [[] for _=1:nbinsϕ]
    n = 1
    for i in orderϕ
        if ϕs[i] > segsϕ[n+1] && !isapprox(ϕs[i],segsϕ[n+1])
            n += 1
        end
        push!(binsϕ[n], i)
    end

    # Place angles in the proper bin.
    angle_ran = [[[] for i=1:nbinsϕ] for j=1:nbinsθ]
    for i = 1:nbinsθ
        for θᵢ in binsθ[i]
            for j = 1:nbinsϕ
                if θᵢ in binsϕ[j]
                    push!(angle_ran[i][j],θᵢ)
                end
            end
        end
    end

    faces = [simplex[:,i] for i=face_ind]
    # Sort points in each bin by distance from the tetrahedron
    for i = 1:nbinsθ
        for j = 1:nbinsϕ
            ptsᵢ = angle_ran[i][j]
            pts = neighbors[:,ptsᵢ]
            order = sortperm([minimum([ptface_mindist(pts[:,i],face) for face = faces]) for i=1:size(pts,2)])
            angle_ran[i][j] = angle_ran[i][j][order]
        end
    end
     
    c = 0
    neighᵢ = []
    while length(neighᵢ) < num_neighbors
        c += 1
        for i=1:nbinsθ
            for j=1:nbinsϕ
                if !(length(angle_ran[i][j]) < c)
                    push!(neighᵢ,neighborsᵢ[angle_ran[i][j][c]])
                end
            end
        end
    end
    neighᵢ             
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
    square. The first four (eight) points are the corners of the square (cube) 
    and need to be disregarded in subsequent computations.

# Examples
```jldoctest
using QHull
import Pebsi.Mesh: ibz_init₋mesh
ibz = chull(Matrix([0. 0. 1. 1.; 0. 1. 0. 1.]'))
n = 2
mesh = ibz_init₋mesh(ibz,n)
mesh.npoints
# output
8
```
"""
function ibz_init₋mesh(ibz::Chull{<:Real},n::Int;
    rtol::Real=sqrt(eps(maximum(ibz.points))),atol::Real=def_atol)::PyObject
    spatial = pyimport("scipy.spatial")

    dim = size(ibz.points,2)
    # We need to enclose the IBZ in a box to prevent collinear triangles.
    box_length = def_mesh_scale*maximum(abs.(ibz.points))
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
    get_sym₋unique!(mesh,pointgroup;rtol,atol)

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
    points of the box). The first k-points, after the first 4, are unique.
- `mesh::PyObject`: a triangulation of a uniform mesh over the IBZ. To avoid
    collinear triangles at the boundary of the IBZ, the IBZ is enclosed in a 
    square. The first four (eight) points are the corners of the square (cube) 
    and need to be disregarded in subsequent computations.

# Examples
```jldoctest
using Pebsi.EPMs: m51
using Pebsi.Mesh: get_sym₋unique!, ibz_init₋mesh
mesh = ibz_init₋mesh(m51.ibz,3)
sym_unique,mesh = get_sym₋unique!(mesh,m51.pointgroup)
sym_unique
# output
12-element Vector{Int64}:
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
  7
```
"""
function get_sym₋unique!(mesh::PyObject,pointgroup::Vector{Matrix{Float64}};
    rtol::Real=sqrt(eps(maximum(mesh.points))),atol::Real=def_atol)

    spatial = pyimport("scipy.spatial")
    dim = size(pointgroup[1],1)
    if dim == 2 nstart = 5 else nstart = 9 end
    # Calculate the unique points of the uniform IBZ mesh.
    n = size(mesh.points,1)
    sym₋unique = zeros(Int,n)
    move = []
    for i=nstart:n
        # If this point hasn't been added already, add it to the list of unique points.
        if sym₋unique[i] == 0
            sym₋unique[i] = i
        else
            push!(move,i)
            continue
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
    # Make the leading points in sym₋unique the unique points in the mesh.
    copy_sym₋unique = deepcopy(sym₋unique)
    if length(move) != 0
        for i = move
            if i == length(move)
                continue
            end 
            for j = i+1:n
                if copy_sym₋unique[j] > i
                    sym₋unique[j] -= 1
                end
            end
        end
        sym₋unique = [zeros(Int,4); sym₋unique[setdiff(5:n,move)]; sym₋unique[move]]
        if move != []
            mesh = spatial.Delaunay([
                mesh.points[1:nstart-1,:];
                mesh.points[setdiff(nstart:n,move),:]; mesh.points[move,:]])
        end
    end
    sym₋unique,mesh
end

@doc """
    notbox_simplices(mesh)

Determine all simplices in a triangulation that do not contain a box point.

# Arguments
- `mesh::PyObject`: a triangulation of a mesh over the IBZ enclosed in a box. It
    is assumed that the first four (eight) points in the mesh are the box points.

# Returns
- `simplicesᵢ::Vector{Vector{Integer}}`: the simplices of the triangulation without
    box points.

# Examples
```jldoctest
using Pebsi.EPMs: m51
using Pebsi.Mesh: ibz_init₋mesh, notbox_simplices
mesh = ibz_init₋mesh(m51.ibz,3)
notbox_simplices(mesh)
# output
8-element Vector{Vector{var"#s29"} where var"#s29"<:Integer}:
 [10, 6, 9]
 [10, 8, 12]
 [8, 10, 9]
 [11, 5, 6]
 [11, 10, 12]
 [10, 11, 6]
 [7, 11, 12]
 [5, 11, 7]
```
"""
function notbox_simplices(mesh::PyObject)::Vector{Vector{<:Integer}}
    simplicesᵢ = [Vector{Int}(mesh.simplices[i,:]) for i=1:size(mesh.simplices,1)]
    dim = size(mesh.points,2)
    if dim == 2 jend = 4 else jend = 8 end
    n = 0
    for i=1:size(mesh.simplices,1)
        # The first few indices are the the corners of a bounding box. Only keep
        # the simplex if it doesn't contain one of these indices.
        if !any([j in (mesh.simplices[i,:] .+ 1) for j=1:jend])
            # Only keep the simplex if it has nonzero volume.
            if !isapprox(simplex_size(Matrix(mesh.points[mesh.simplices[i,:] .+ 1,:]')),0,atol=def_atol)
                n += 1
                simplicesᵢ[n] = mesh.simplices[i,:] .+ 1
            end
        end
    end
    simplicesᵢ[1:n]
end

@doc """
    get_cvpts(mesh,ibz,atol)

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
using Pebsi.QuadraticIntegration: get_cvpts
using PyCall; spatial = pyimport("scipy.spatial")
using QHull
ibz = chull(Matrix([0. 0. 1. 1.; 0. 1. 0. 1.]'))
pts = [0. 0. 0. 0.5 0.5 0.5 1.0 1.0 1.0; 0. 0.5 1. 0. 0.5 1. 0. 0.5 1.]
mesh = spatial.Delaunay(Matrix(pts'))
get_cvpts(mesh,ibz)
# output
8-element Vector{Int64}:
 1
 2
 3
 4
 6
 7
 8
 9
```
"""
function get_cvpts(mesh::PyObject,ibz::Chull;atol::Real=def_atol)::AbstractVector{<:Int}
    dim = size(mesh.points,2)
    if dim == 2
        borders = [Matrix(ibz.points[i,:]') for i=ibz.simplices]
        distfun = lineseg₋pt_dist
    else
        borders = [Matrix(ibz.points[f,:]') for f=get_uniquefacets(ibz)]
        distfun = ptface_mindist
    end

    cv_pointsᵢ = [0 for i=1:size(mesh.points,1)]
    n = 0
    for i=1:size(mesh.points,1)
        if any([isapprox(distfun(mesh.points[i,:],border),0,atol=atol) 
            for border=borders])
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
```
using Pebsi.EPMs: m21
using Pebsi.Mesh: ibz_init₋mesh, get_extmesh
mesh = ibz_init₋mesh(m21.ibz,1)
mesh,extmesh,sym_unique = get_extmesh(m21.ibz,mesh,m21.pointgroup,m21.recip_latvecs)
```
"""
function get_extmesh(ibz::Chull,mesh::PyObject,pointgroup::Vector{Matrix{Float64}},
    recip_latvecs::AbstractMatrix{<:Real},near_neigh::Int=1;
    rtol::Real=sqrt(eps(maximum(abs.(mesh.points)))),atol::Real=def_atol)
    
    dim = size(recip_latvecs,1)
    spatial = pyimport("scipy.spatial")
    sym₋unique,mesh = get_sym₋unique!(mesh,pointgroup)
    cv_pointsᵢ = get_cvpts(mesh,ibz)

    # Calculate the maximum distance between neighboring points
    bound_limit = def_max_neighbor_tol*maximum(
        reduce(vcat,[[norm(mesh.points[i,:] - mesh.points[j,:]) 
                    for j=get_neighbors(i,mesh,near_neigh)] for i=cv_pointsᵢ]))

    if dim == 2
        borders = [Matrix(ibz.points[i,:]') for i=ibz.simplices]
        distfun = lineseg₋pt_dist
        bztrans = [[[i,j] for i=-1:1,j=-1:1]...]
    else
        borders = [Matrix(ibz.points[f,:]') for f=get_uniquefacets(ibz)]
        distfun = ptface_mindist
        bztrans = [[[i,j,k] for i=-1:1,j=-1:1,k=-1:1]...]
    end 

    neighborsᵢ = reduce(vcat,[get_neighbors(i,mesh,near_neigh) for i=cv_pointsᵢ]) |> unique
    neighbors = zeros(Float64,dim,length(neighborsᵢ)*length(pointgroup)*length(bztrans));
    sym₋unique = [sym₋unique; zeros(Int,size(neighbors,2))];
    numpts = size(mesh.points,1)    
    n = 0
    for i=neighborsᵢ,op=pointgroup,trans=bztrans
        pt = op*mesh.points[i,:] + recip_latvecs*trans
        if any([distfun(pt,border) < bound_limit for border=borders]) &&
            !any(mapslices(x->isapprox(x,pt,atol=atol,rtol=rtol),[mesh.points' neighbors[:,1:n]],dims=1))
            n += 1
            neighbors[:,n] = pt
            sym₋unique[numpts + n] = sym₋unique[i]
        end
    end

    neighbors = neighbors[:,1:n]
    sym₋unique = sym₋unique[1:numpts + n]
    ext_mesh = spatial.Delaunay(unique_points([mesh.points; neighbors']',
        rtol=rtol,atol=atol)')
    
    (mesh,ext_mesh,sym₋unique)
end

"""
    trimesh(ndivs)

Split a triangle uniformly into smaller triangles and sample each subtriangle at its center.

# Arguments
- `ndivs::Integer`: the number of divisions along one side of the triangle.

# Returns
- `mesh::Matrix{<:Real}`: the points in the mesh in barycentric coordinates in
    columns of a matrix.

# Examples
```jldoctest
using Pebsi.Mesh: trimesh
trimesh(2)
# output
3×4 Matrix{Float64}:
 0.666667  0.333333  0.166667  0.166667
 0.166667  0.333333  0.666667  0.166667
 0.166667  0.333333  0.166667  0.666667
```
"""
function trimesh(ndivs::Integer)::Matrix{<:Real}
    dim = 2
    bpts = sample_simplex(dim,ndivs) 
    mesh = zeros(3,ndivs*ndivs)
    r0,r1 = 0,0 # row offset
    n = ndivs+1
    counter = 0
    for j=1:ndivs
        for i=1:ndivs+1-j
            counter += 1
            r0,r1 = sum(n:-1:n-j+2),sum(n:-1:n-j+1)
            # Take the average of the point at the corners of the subtriangle      
            mesh[:,counter] = mean(bpts[:,[i+r0,i+r0+1,i+r1]],dims=2)
            if i < ndivs-j+1 && j != ndivs
                counter += 1
                # Add two triangles if not at the boundary
                mesh[:,counter] = mean(bpts[:,[i+r0+1,i+1+r1,i+r1]],dims=2)
            end
        end
    end
    mesh
end

end # Module