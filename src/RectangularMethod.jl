module RectangularMethod


import SymmetryReduceBZ.Symmetry: mapto_unitcell
import Pebsi.EPMs: eval_EPM
import Base.Iterators: product
import LinearAlgebra: det, diag
import AbstractAlgebra: ZZ, matrix, snf_with_transform, hnf_with_transform, hnf


"""
    sample_unitcell(latvecs,N,grid_offset,rtol,atol)

Create a generalized regular grid over the unit cell.

# Arguments
- `latvecs::AbstractArray{<:Real,2}`: the reciprocal lattice vectors as columns
    of a square array.
- `N::AbstractArray{<:Integer,2}`: an integer, square array that relates the
    reciprocal lattice vectors `R` to the grid generating vectors `K`: `R=KN`.
- `grid_offset::AbstractArray{<:Real,1}=[0,0]`: the offset of the grid in grid
    coordinates (fractions of the grid generating vectors).
- `rtol::Real=sqrt(eps(float(maximum(latvecs))))`: a relative tolerance for
    floating point comparisons. This is used for mapping points into the
    provided unit cell.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `::Array{Float64,2}`: the points in the generalized, regular grid as
    columns of a 2D array.
# Examples
```jldoctest
recip_latvecs = [1 0; 0 1]
N = [2 0; 0 2]
grid_offset = [0.5, 0.5]
sample_unitcell(recip_latvecs,N,grid_offset)
# output
2×4 Array{Real,2}:
 0.25  0.75  0.25  0.75
 0.25  0.25  0.75  0.75
```
"""
function sample_unitcell(latvecs::AbstractArray{<:Real,2},
    N::AbstractArray{<:Integer,2},
    grid_offset::AbstractArray{<:Real,1}=[0,0],
    rtol::Real=sqrt(eps(float(maximum(latvecs)))),
    atol::Real=1e-9)::Array{Float64,2}

    H = hnf(matrix(ZZ,N))
    H = convert(Array{Integer,2},Array(H))
    gridvecs = inv(H)*latvecs
    inv_latvecs = inv(latvecs)
    if size(latvecs) == (2,2) && length(grid_offset) == 2
        (a,c) = diag(H)
        grid = reduce(hcat,[gridvecs*([i,j] + grid_offset) for
            (i,j)=product(0:a-1,0:c-1)])
    elseif size(latvecs) == (3,3) && length(grid_offset) == 3
        (a,c,f) = diag(H)
        grid = reduce(hcat,[gridvecs*([i,j,k] + grid_offset) for
            (i,j,k)=product(0:a-1,0:c-1,0:f-1)])
    else
        throw(ArgumentError("The lattice vectors and offset are incompatible."))
    end

    reduce(hcat,[mapto_unitcell(grid[:,i],latvecs,inv_latvecs,"Cartesian",
        rtol,atol) for i=1:size(grid,2)])
end

"""
    rectangular_method(real_latvecs,atom_types,atom_pos,rules,cutoff,
        sheets,ndivs,grid_offset)

Calculate the Fermi level and band energy with the rectangular method.
"""
function rectangular_method(recip_latvecs,atom_types,atom_pos,rules,electrons,
    cutoff,sheets,ndivs,grid_offset=[0,0])

    integration_points = sample_unitcell(recip_latvecs, ndivs, grid_offset)
    num_kpoints = size(integration_points,2)
    num_states = num_kpoints*sheets[end]
    eigenvalues = zeros(num_states)

    for i=1:num_kpoints
        eigenvalues[1+(i-1)*sheets[end]:(sheets[end]*i)] = eval_EPM(
            integration_points[:,i], recip_latvecs, rules, cutoff,sheets)
    end

    sort!(eigenvalues);
    occupied_states = 1:ceil(Int,electrons*num_kpoints/2)
    rectangle_size = det(recip_latvecs)/num_kpoints
    fermi_level = eigenvalues[occupied_states[end]]
    band_energy = rectangle_size*sum(eigenvalues[occupied_states])

    (num_kpoints,fermi_level,band_energy)
end


"""
    symreduce_grid(recip_latvecs,ndivs,grid_offset,pointgroup,rtol,atol)

Calculate the symmetrically unique points and their weights in a GR grid.

# Arguments
- `recip_latvecs::AbstractArray{<:Real,2}`:
- `N::AbstractArray{Integer,2}`: the integer matrix that relates the reciprocal
    lattice vectors (R) to the grid generating vectors (K): `K = R*N`.
- `grid_offset::AbstractArray{<:Real,1}`: the offset of the grid in grid
    coordinates or fractions of the grid generating vectors (K). The offset in
    Cartesian coordinates is `K*gridoffset`.
- `pointgroup::AbstractArray`: the operators in the point group in matrix
    representation. They operate on points in Cartesian coordinates.
- `rtol::Real=sqrt(eps(float(maximum(recip_latvecs))))`: relative tolerance for
    floating point comparisons. This is needed when mapping points into the
    first unit cell.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons. Also
    used for mapping points to the first unit cell.

# Returns
- `kpoint_weights::Array{Int64,1}`: a list weights or sizes of orbits for the
    symmetrically distinct points in the grid.
- `unique_kpoints::Array{Float64,2}`: the representative points for each orbit
    as columns of a 2D array in Cartesian coordinates.
- `orbits::Array{Array{Float64,2},1}`(optional): a list of points in each orbit.

# Examples
```jldoctest
recip_latvecs = [1 0; 0 1]
N = [2 0; 0 2]
grid_offset = [0.5, 0.5]
pointgroup = [[0.0 -1.0; -1.0 0.0], [0.0 -1.0; 1.0 0.0], [-1.0 0.0; 0.0 -1.0],
    [1.0 0.0; 0.0 -1.0], [-1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0],
    [0.0 1.0; -1.0 0.0], [0.0 1.0; 1.0 0.0]]
symreduce_grid(recip_latvecs,N,grid_offset,pointgroup)
# output
([4], [0.25; 0.25])
```
"""
function symreduce_grid(recip_latvecs::AbstractArray{<:Real,2},
    N::AbstractArray{<:Integer,2}, grid_offset::AbstractArray{<:Real,1},
    pointgroup::AbstractArray, include_orbits::Bool=false,
    rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),
    atol::Real=1e-9)

    if (size(N) == (2,2) && size(recip_latvecs) == (2,2) &&
            length(grid_offset) == 2)
        dim = 2
        kpoint_index = kpoint_index2D
    elseif (size(N) == (3,3) && size(recip_latvecs) == (3,3) &&
            length(grid_offset) == 3)
        dim = 3
        kpoint_index = kpoint_index3D
    else
        throw(ArgumentError("Only regular grids in 2D or 3D can be reduced."))
    end

    origin = zeros(Int,dim)
    inv_rlatvecs = inv(recip_latvecs)
    K = inv(N)*recip_latvecs
    invK = inv_rlatvecs*N
    offset = K*grid_offset
    N = matrix(ZZ,N)
    (H,U) = hnf_with_transform(N)
    (D,A,B) = snf_with_transform(H)
    (D,A,B,N) = [Array(x) for x=[D,A,B,N]]
    (D,A,B,N) = [convert(Array{Int64,2},x) for x=[D,A,B,N]]
    d = diag(D)
    grid = sample_unitcell(recip_latvecs,N,grid_offset,rtol,atol)
    num_kpoints = size(grid,2)
    unique_count = 0

    # hash_table keeps track of k-points within orbits
    hash_table = zeros(Int,num_kpoints)
    # The representative k-point for each orbit
    first_kpoint = zeros(Int,num_kpoints)
    kpoint_weights = zeros(Int,num_kpoints)
    # A list of indices, used as a failsafe
    indices = zeros(Int,num_kpoints)

    for i=1:num_kpoints
        # Remove the offset because indexing doesn't work with shifted grids.
        dialᵢ = mod.(A*round.(Int,invK*(grid[:,i]-offset)),d)
        indexᵢ = kpoint_index(dialᵢ,d)
        indices[i] = indexᵢ
        # Move on if this k-point is already part of an orbit.
        if hash_table[indexᵢ] != 0
            continue
        end
        unique_count += 1
        hash_table[indexᵢ] = unique_count
        first_kpoint[indexᵢ] = i
        kpoint_weights[indexᵢ] = 1
        for (j,op) in enumerate(pointgroup)
            test = op*grid[:,i]

            rot_point = mapto_unitcell(op*grid[:,i],recip_latvecs,inv_rlatvecs,
                "Cartesian",rtol,atol)
            test = mapto_unitcell(rot_point-offset,K,invK,"Cartesian",rtol,atol)
            if !(isapprox(test,origin,rtol=rtol,atol=atol))
                continue
            end
            dialⱼ = mod.(A*round.(Int,invK*(rot_point-offset)),d)
            indexⱼ = kpoint_index(dialⱼ,d)
            if hash_table[indexⱼ] == 0
                hash_table[indexⱼ] = unique_count
                kpoint_weights[indexᵢ] += 1
            end
        end
    end
    @show hash_table
    test = sort(indices) - collect(1:num_kpoints)
    test = findall(x->x!=0,test)
    if sort(indices) != collect(1:num_kpoints)
        error("The k-point indices are calculated incorrectly.")
    end

    indices = findall(x->x!=0,kpoint_weights)
    unique_kpoints = grid[:,first_kpoint[indices]]
    kpoint_weights = kpoint_weights[indices]

    if include_orbits
        orbits = Array{Array{Float64,2},1}([])
        for i=1:unique_count
            append!(orbits,[grid[:,findall(x->x==i,hash_table)]])
        end
        (kpoint_weights, unique_kpoints, orbits)
    else
        (kpoint_weights, unique_kpoints)
    end
end


"""
    kpoint_index2D(dial, snf_diag)

Calculate the index of a k-point in a regular grid in 2D.

# Arguments
- `dial::AbstractArray{Integer,1}`: the odometer reading of the point in the
    generalized regular grid.
- `snf_diag::AbstractArray{Integer,1}`: the diagonal elements of the integer
    matrix that relates the lattice vectors to the superlattice vectors in
    Smith normal form.

# Returns
- `::Integer`: the index of the k-point in the grid.

# Examples
```jldoctest
import Pebsi.RectangularMethod: kpoint_index2D
dial = [1,2]
snf_diag = [2,3]
kpoint_index2D(dial,snf_diag)
# output
6
```
"""
function kpoint_index2D(dial::AbstractArray{<:Integer,1},
    snf_diag::AbstractArray{<:Integer,1})::Integer
    round(Int,1 + dial[2] + dial[1]*snf_diag[2])
end

"""
    kpoint_index3D(dial, snf_diag)

Calculate the index of a k-point in a regular grid in 3D.

# Arguments
- `dial::AbstractArray{Integer,1}`: the odometer reading of the point in the
    generalized regular grid.
- `snf_diag::AbstractArray{Integer,1}`: the diagonal elements of the integer
    matrix that relates the lattice vectors to the superlattice vectors in
    Smith normal form.

# Returns
- `::Integer`: the index of the k-point in the grid.

# Examples
```jldoctest
import Pebsi.RectangularMethod: kpoint_index3D
dial = [1,2,0]
snf_diag = [2,4,8]
kpoint_index3D(dial,snf_diag)
# output
49
```
"""
function kpoint_index3D(dial::AbstractArray{<:Integer,1},
    snf_diag::AbstractArray{<:Integer,1})::Integer
    round(Int,1 + dial[3] + dial[2]*snf_diag[3] + dial[1]*
        snf_diag[3]*snf_diag[2])
end


"""
    calculate_orbits(grid,pointgroup,latvecs)

Calculate the points of the grid in each orbit the hard way.

# Arguments
- `grid::AbstractArray{<:Real,2}`: the points in the grid as columns of an
    array in Cartesian coordinates.
- `pointgroup::AbstractArrray`: the operators in the point group in nested
    array. They operator on points in Cartesian coordinates.
- `latvecs::AbstractArray{<:Real,2}`: the lattice vectors as columns of an
    array.

# Returns
- `orbits::AbstractArray`: the points of the grid in each orbit in a nested
    array.

# Examples
```jldoctest
import Pebsi.RectangularMethod: calculate_orbits, sample_unitcell
import SymmetryReduceBZ.Symmetry: calc_pointgroup
recip_latvecs = [1 0; 0 1]
N = [2 0; 0 2]
grid_offset = [0.5, 0.5]
pointgroup = calc_pointgroup(recip_latvecs)

grid = sample_unitcell(recip_latvecs,N,grid_offset)
calculate_orbits(grid,pointgroup,recip_latvecs)
# output
1-element Array{Array{Float64,2},1}:
 [0.75 0.75 0.25 0.25; 0.75 0.25 0.75 0.25]
```
"""
function calculate_orbits(grid::AbstractArray{<:Real,2},
    pointgroup,latvecs::AbstractArray{<:Real,2},
    rtol::Real=sqrt(eps(float(maximum(grid)))),atol::Real=1e-9)::AbstractArray
    inv_latvecs = inv(latvecs)
    coordinates = "Cartesian"
    uc_grid = reduce(hcat,[mapto_unitcell(grid[:,i],latvecs,inv_latvecs,
        coordinates) for i=1:size(grid,2)]);
    numpts = 0
    orbits = [zeros(2,length(pointgroup)) for i=1:size(uc_grid,2)]
    orbit = zeros(2,length(pointgroup))
    gridcopy =  uc_grid
    while size(gridcopy) != (2,0)
        pt = gridcopy[:,1]
        numpts += 1
        orbit = zeros(2,length(pointgroup))
        orbsize = 0
        for op in pointgroup
            rot_pt = Array{Float64,1}(op*pt)
            rot_pt = mapto_unitcell(rot_pt,latvecs,inv_latvecs,coordinates)
            pos = sort(unique([isapprox(rot_pt,gridcopy[:,i],rtol=rtol,
                atol=atol) ? i : 0 for i=1:size(gridcopy,2)]))[end]
            if pos !=0
                orbsize += 1
                orbit[:,orbsize] = gridcopy[:,pos]
                gridcopy = gridcopy[:,1:end .!= pos]
                if size(gridcopy) == (2,0)
                    break
                end
            end
        end
        orbit = orbit[:,1:orbsize]
        orbits[numpts] = orbit
    end
    orbits = orbits[1:numpts]
end

end # module
