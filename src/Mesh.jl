module Mesh

using ..Geometry: simplex_size, barytocart, lineseg₋pt_dist, ptface_mindist,
    sample_simplex
using ..Defaults: def_atol, def_mesh_scale, def_max_neighbor_tol,
    def_neighbors_per_bin2D, def_neighbors_per_bin3D, def_num_neighbors2D, 
    def_num_neighbors3D, def_kpoint_tol
using SymmetryReduceBZ.Utilities: unique_points, get_uniquefacets, sortpts2D
using PyCall: pyimport,PyObject
using QHull: Chull
using Statistics: mean
using LinearAlgebra: norm, dot, cross
using Suppressor

export get_neighbors, choose_neighbors, choose_neighbors3D, ibz_init₋mesh, 
    get_sym₋unique!, notbox_simplices, get_cvpts, get_extmesh, trimesh, ntripts,
    ntetpts, simplex_cornerpts, gmsh_initmesh, ibz_initmesh

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
- `n::Int`: a measure of the number of k-points in the mesh that is the number of 
    divisions of an edge of the IBZ when the IBZ is a simplex or 
- `rtol::Real=sqrt(eps(maximum(ibz.points)))`: a relative tolerance for finite
    precision comparisons.
- `atol::Real=1e-9`: an absolute tolerance for finite precision comparisons.

# Returns
- `mesh::PyObject`: a triangulation of a uniform mesh over the IBZ. To avoid
    collinear triangles at the boundary of the IBZ, the IBZ is enclosed in a 
    box. The first four (eight) points are the corners of the square (cube) 
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
13
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
    simplices = [Array(mesh.points[mesh.simplices[i,:].+1,:]') for i=1:size(mesh.simplices,1)]
    simplex_sizes = [simplex_size(s) for s=simplices]
    simplex_divs = n*(simplex_sizes ./ minimum(simplex_sizes)).^(1/dim)
    simplex_divs = [mod(d,n) == 0 ? Int(d) : round(Int,d/n)*n for d=simplex_divs]
    pts = unique_points(reduce(hcat,[barytocart(sample_simplex(
        dim,simplex_divs[i]),simplices[i]) for i=1:length(simplex_divs)]),atol=atol,rtol=rtol)
    mesh = spatial.Delaunay([box_pts'; pts'])
end

@doc """
    get_sym₋unique!(points,pointgroup;rtol,atol)

Calculate the symmetrically unique points within the IBZ.

# Arguments
- `points::Matrix{<:Real}`: a triangulation of a mesh over the IBZ.
- `pointgroup::Vector{Matrix{Float64}}`: the point operators of the real-space
    lattice. They operate on points in Cartesian coordinates.
- `rtol::Real=sqrt(eps(maximum(real_latvecs)))`: a relative tolerance.
- `atol::Real=1e-9`: an absolute tolerance.

# Returns
- `sym₋unique::AbstractVector{<:Int}`: a vector that gives the position of the k-point
    that is equivalent to each k-point (except for the first 4 points or the
    points of the box). The first k-points, after the first 4, are unique.
- `points::PyObject`: The points in the mesh, with the unique points first. To avoid
    collinear triangles at the boundary of the IBZ, the IBZ is enclosed in a 
    square. The first four (eight) points are the corners of the square (cube) 
    and need to be disregarded in subsequent computations.

# Examples
```jldoctest
using Pebsi.EPMs: m51
using Pebsi.Mesh: get_sym₋unique!, ibz_init₋mesh
mesh = ibz_init₋mesh(m51.ibz,3)
sym_unique,points = get_sym₋unique!(Matrix(mesh.points'),m51.pointgroup)
length(sym_unique)
# output
44
```
"""
function get_sym₋unique!(points::Matrix{<:Real},pointgroup::Vector{Matrix{Float64}};
    cvpts::Union{Nothing,Vector{<:Integer}}=nothing,
    rtol::Real=sqrt(eps(maximum(points))),atol::Real=def_atol) 
    spatial = pyimport("scipy.spatial")
    dim = size(pointgroup[1],1)
    if dim == 2 nstart = 5 else nstart = 9 end
    # Calculate the unique points of the uniform IBZ mesh.
    n = size(points,2)
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
        if cvpts != nothing
            if !(i in cvpts)
                continue
            end
        end
        for pg=pointgroup
            rotpts = [mapslices(x->isapprox(x,pg*points[:,i],atol=atol,
                rtol=rtol),points,dims=1)...]
            pos = findall(x->x==1,rotpts)
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
        sym₋unique = [zeros(Int,nstart-1); sym₋unique[setdiff(nstart:n,move)]; sym₋unique[move]]
        if move != []
            points = [points[:,1:nstart-1] points[:,setdiff(nstart:n,move)] points[:,move]] 
        end
    end
    sym₋unique,points
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
```
using Pebsi.EPMs: m51
using Pebsi.Mesh: ibz_init₋mesh, notbox_simplices
mesh = ibz_init₋mesh(m51.ibz,1)
notbox_simplices(mesh)
# output
8-element Vector{Vector{var"#s18"} where var"#s18"<:Integer}:
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
    get_cvpts(points,ibz,atol)

Determine which points are on the boundary of the IBZ (or any convex hull).

# Arguments
- `points::Matrix{<:Real}`: a matrix of Cartesian points as columns of a matrix.
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
get_cvpts(Matrix(mesh.points'),ibz)
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
function get_cvpts(points::Matrix{<:Real},ibz::Chull;atol::Real=def_atol)::AbstractVector{<:Integer}
    dim = size(points,1)
    if dim == 2
        borders = [Matrix(ibz.points[i,:]') for i=ibz.simplices]
        distfun = lineseg₋pt_dist
    else
        borders = [Matrix(ibz.points[f,:]') for f=get_uniquefacets(ibz)]
        distfun = ptface_mindist
    end
    cv_pointsᵢ = [0 for i=1:size(points,2)]
    n = 0
    for i=1:size(points,2)
        if any([isapprox(distfun(points[:,i],border),0,atol=atol) 
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
    cv_pointsᵢ = get_cvpts(Matrix(mesh.points'),ibz)
    sym₋unique,points = get_sym₋unique!(Matrix(mesh.points'), pointgroup, cvpts=cv_pointsᵢ)
    mesh = spatial.Delaunay(points')
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
            # Take the average of the points at the corners of the subtriangle      
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

"""Gives the number of points over a triangle for a number of divisions"""
ntripts(n) = sum(1:n+1)
"""Gives the number of triangles for a number of divisions"""
ntriangles(n) = n^2

"""Gives the number of points over a tetrahedron for a number of divisions"""
ntetpts(n) = sum([sum(1:i) for i=1:n+1])
"""Gives the number of tetrahedra for a number of divisions"""
ntetrahedra(n) = n^3

@doc """
    simplex_cornerpts(dim,deg)

Locate the indices of corners of a simplex from the Bezier points.

# Arguments
- `dim::Integer`: the dimension of the space.
- `deg::Integer`: the degree of the polynomial.

# Returns
- `Indices::Vector{<:Integer}`: the indices of the corners points

# Examples
```jldoctest
using Pebsi.Mesh
simplex_cornerpts(3,4)
# output
4-element Vector{Int64}:
  1
  5
 15
 35
```
"""
function simplex_cornerpts(dim::Integer,deg::Integer)
    if dim > 3
        error("Valid up to three dimensions.")
    elseif deg < 1
        error("Valid for polynomial degree of at least 1.")
    end
    indices = [1,deg+1]
    nptfuns = [ntripts,ntetpts]
    if dim > 1
        for di=2:dim
            push!(indices,nptfuns[di-1](deg))
        end
    end
    indices    
end

@doc """
    gmsh_initmesh(ibz,meshsize;opt_threshold,mesh_algo,mesh_algo3D,
        opt_algo, opt_iters,atol)

Calculate a uniform mesh over the IBZ using gmsh.

# Arguments
- `ibz::Chull`: 
- `meshsize::Real`:
- `opt_threshold::Real=1.0`: 
- `mesh_algo::Integer=6`: 
- `mesh_algo3D::Integer=1`: 
- `opt_algo::Integer=1`: 
- `opt_iters::Integer=100`:
- `atol::Real=def_atol`: 

# Returns
- `mesh`: a triangular or tetrahedral mesh over the IBZ
- `msimplicesᵢ`: the simplices of the mesh obtained from gmsh.

# Examples
```
using Pebsi.EPMs, Pebsi.Mesh
ibz = epm.m11.ibz
meshsize = 0.1
gmsh_initmesh(ibz,meshsize)
```
"""
function gmsh_initmesh(ibz::Chull, meshsize::Real;
    opt_threshold::Real=1.0, mesh_algo::Integer=6, mesh_algo3D::Integer=1,
    opt_algo::Integer=1, opt_iters::Integer=100, atol::Real=def_atol)
    gmsh = pyimport("gmsh")
    spatial = pyimport("scipy.spatial")
    ibzpts = Matrix(ibz.points')
    n = size(ibzpts,2); dim = size(ibzpts,1)
    if dim == 2
        ibzpts = ibzpts[:,sortpts2D(ibzpts)]
        ibzpts = [ibzpts; zeros(size(ibzpts,2))']
    end
    gmsh.initialize()
    gmsh.model.add("ibz")
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin",meshsize)
    gmsh.option.setNumber("Mesh.MeshSizeMax",meshsize)
    gmsh.option.setNumber("Mesh.Optimize", opt_algo)
    gmsh.option.setNumber("Mesh.OptimizeThreshold", opt_threshold)
    gmsh.option.setNumber("Mesh.Algorithm", mesh_algo)
    gmsh.option.setNumber("Mesh.Algorithm3D", mesh_algo3D)
    for i=1:n
        gmsh.model.geo.addPoint(ibzpts[:,i]...,meshsize,i)
    end
    
    if dim == 2
        for i=1:n
            gmsh.model.geo.addLine(i, mod1(i+1,n), i)
        end
        gmsh.model.geo.addCurveLoop(collect(1:n), n+1)
        gmsh.model.geo.addPlaneSurface([n+1], n+2)
    else
        # Make normals to surface point away from the IBZ
        indices = get_uniquefacets(ibz)
        c = mean(ibzpts,dims=2)
        for (i,ind) in enumerate(indices)
            p1,p2,p3 = [ibzpts[:,i] for i=ind[1:3]]
            nor = cross(p2-p1, p3 - p1)
            d = [dot(nor,p-c) for p=[p1,p2,p3]]
            if all(d .> 0 .| isapprox.(d,0,atol=atol))
                indices[i] = reverse(ind)
            end
        end
        f = n+1
        sl = []
        edges = []
        edgesᵢ = []
        for (i,ind) in enumerate(indices)
            m = length(ind)
            ls = []
            for j=1:m
                edge = (ind[j], ind[mod1(j+1,m)])
                pos = findall(x->(x==edge)||x==reverse(edge),edges)
                if pos == []
                    gmsh.model.geo.addLine(edge..., f+j)
                    edges = [edges; edge]
                    edgesᵢ = [edgesᵢ; f+j]
                    ls = [ls;f+j]
                else
                    if edge in edges
                        ls = [ls;edgesᵢ[pos[1]]]
                    else
                       ls = [ls;-edgesᵢ[pos[1]]]
                    end
                end
            end
            gmsh.model.geo.addCurveLoop(ls, f+m+1)
            gmsh.model.geo.addPlaneSurface([f+m+1], f+m+2)
            sl = [sl;f+m+2]
            f += m + 2
        end
        gmsh.model.geo.addSurfaceLoop(sl,f+1)
        gmsh.model.geo.addVolume([f+1],f+2)
    end 
    gmsh.model.geo.synchronize()
    gmsh.model.mesh.generate(dim=dim)
    entities = gmsh.model.getEntities()
    pts = []
    for ent in entities
        nt,nc,npa = gmsh.model.mesh.getNodes(ent...)
        pts = [pts; nc]
    end
    pts = reshape(pts,3,Int(length(pts)/3))
    pts = pts[1:dim,:]
    box_length = def_mesh_scale*maximum(abs.(ibz.points))
    if dim == 2
        box_pts = reduce(hcat,[[mean(ibz.points,dims=1)...] + box_length*[i,j] 
            for i=[-1,1] for j=[-1,1]])
    elseif dim == 3
        box_pts = reduce(hcat,[[mean(ibz.points,dims=1)...] + box_length*[i,j,k] 
            for i=[-1,1] for j=[-1,1] for k=[-1,1]])  
    end
    elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(dim)
    nt = Int.(elemNodeTags[1])
    simplicesᵢ = [nt[i:i+dim] .+ 2^dim for i=1:dim+1:length(nt)-dim];
    mesh = spatial.Delaunay([box_pts pts]')
    gmsh.finalize()
    mesh,simplicesᵢ
end

@doc """
    ibz_initmesh(ibz,num_kpoints;opt_threshold,mesh_algo,mesh_algo3D,
        opt_algo, opt_iters,atol)

Calculate a uniform mesh over the IBZ using gmsh.

# Arguments
- `ibz::Chull`: 
- `meshsize::Real`:
- `opt_threshold::Real=1.0`: 
- `mesh_algo::Integer=6`: 
- `mesh_algo3D::Integer=1`: 
- `opt_algo::Integer=1`: 
- `opt_iters::Integer=100`:
- `atol::Real=def_atol`: 

# Returns
- `mesh`: a triangular or tetrahedral mesh over the IBZ
- `msimplicesᵢ`: the simplices of the mesh obtained from gmsh.

# Examples
```
using Pebsi.EPMs, Pebsi.Mesh
ibz = epm.m11.ibz
meshsize = 0.1
gmsh_initmesh(ibz,meshsize)
```
"""
function ibz_initmesh(ibz,num_kpoints;
    opt_threshold::Real=1.0, mesh_algo::Integer=6, mesh_algo3D::Integer=1,
    opt_algo::Integer=1, opt_iters::Integer=100, kpoint_tol::Real=def_kpoint_tol, 
    atol::Real=def_atol)
    dim = size(ibz.points,2)
    # Estimate the size of the mesh that will give the desired number of k-points
    meshsize = (ibz.volume/num_kpoints)^(1/dim)
    # Find mesh sizes that have more and less than the desired number of k-points
    meshsize₀ = meshsize; meshsize₁ = meshsize
    @suppress mesh,simplices = gmsh_initmesh(ibz, meshsize, opt_threshold=opt_threshold,
    mesh_algo=mesh_algo, mesh_algo3D=mesh_algo3D, opt_algo=opt_algo, opt_iters=opt_iters,
    atol=atol)
    mesh₀ = mesh; mesh₁ = mesh; simplices₀ = simplices; simplices₁ = simplices
    npts = size(mesh.points,1) - 2^dim
    npts₀ = npts; npts₁ = npts
    if npts > num_kpoints
        while npts₀ > num_kpoints
            meshsize₀ *= 2
            @suppress mesh₀,simplices₀ = gmsh_initmesh(ibz, meshsize₀, opt_threshold=opt_threshold,
            mesh_algo=mesh_algo, mesh_algo3D=mesh_algo3D, opt_algo=opt_algo, opt_iters=opt_iters,
            atol=atol)
            npts₀ = size(mesh₀.points,1) - 2^dim
        end
    else
        while npts₁ < num_kpoints
            meshsize₁ /= 2
            @suppress mesh₁,simplices₁ = gmsh_initmesh(ibz, meshsize₁, opt_threshold=opt_threshold,
            mesh_algo=mesh_algo, mesh_algo3D=mesh_algo3D, opt_algo=opt_algo, opt_iters=opt_iters,
            atol=atol)
            npts₁ = size(mesh₁.points,1) - 2^dim
        end
    end
    meshsize = (meshsize₀ + meshsize₁)/2
    mesh,simplices = gmsh_initmesh(ibz, meshsize)
    npts = size(mesh.points,1) - 2^dim
    prev_npts = npts
    while abs(npts - num_kpoints) > kpoint_tol*num_kpoints
        if npts > num_kpoints
            npts₁ = npts; meshsize₁ = meshsize
        else
            npts₀ = npts; meshsize₀ = meshsize
        end
        meshsize = (meshsize₀ + meshsize₁)/2
        @suppress mesh,simplices = gmsh_initmesh(ibz, meshsize, opt_threshold=opt_threshold,
        mesh_algo=mesh_algo, mesh_algo3D=mesh_algo3D, opt_algo=opt_algo, opt_iters=opt_iters,
        atol=atol)
        npts = size(mesh.points,1) - 2^dim
        if prev_npts == npts
            break
        else
            prev_npts = npts
        end
    end
    mesh,simplices
end
end # Module