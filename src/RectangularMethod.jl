module RectangularMethod

using SymmetryReduceBZ.Symmetry: mapto_unitcell, mapto_bz
using Base.Iterators: product
using LinearAlgebra: det, diag, dot, inv
using AbstractAlgebra: ZZ, matrix, snf_with_transform, hnf_with_transform, hnf
using Distributed: pmap

using ..EPMs: eval_epm, RytoeV, epm₋model, epm₋model2D, sym_offset
using ..Defaults: def_atol

export sample_unitcell, rectangular_method, symreduce_grid, convert_mixedradix,
    kpoint_index, calculate_orbits

@doc """
    sample_unitcell(latvecs,N,grid_offset,rtol,atol)

Create a generalized regular grid over the unit cell.

# Arguments
- `latvecs::AbstractMatrix{<:Real}`: the reciprocal lattice vectors as columns
    of a square matrix.
- `N::AbstractMatrix{<:Integer}`: an integer, square matrix that relates the
    reciprocal lattice vectors `R` to the grid generating vectors `K`: `R=KN`.
- `grid_offset::AbstractVector{<:Real}=zeros(size(N,1))`: the offset of the grid
    in grid coordinates (fractions of the grid generating vectors).
- `rtol::Real=sqrt(eps(float(maximum(latvecs))))`: a relative tolerance for
    floating point comparisons. This is used for mapping points into the
    provided unit cell.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `::Matrix{Float64}`: the points in the generalized regular grid as columns of a 2D array.

# Examples
```jldoctest
using Pebsi.RectangularMethod: sample_unitcell
recip_latvecs = [1 0; 0 1]
N = [2 0; 0 2]
grid_offset = [0.5, 0.5]
sample_unitcell(recip_latvecs,N,grid_offset)
# output
2×4 Matrix{Float64}:
 0.25  0.75  0.25  0.75
 0.25  0.25  0.75  0.75
```
"""
function sample_unitcell(latvecs::AbstractMatrix{<:Real},
    N::AbstractMatrix{<:Integer},
    grid_offset::AbstractVector{<:Real}=zeros(size(N,1));
    rtol::Real=sqrt(eps(float(maximum(latvecs)))),
    atol::Real=def_atol)::AbstractMatrix{Float64}

    H = hnf(matrix(ZZ,N))
    H = convert(Array{Int,2},Array(H))
    gridvecs = inv(H)*latvecs
    inv_latvecs = inv(latvecs)

    offset = (inv(H)*latvecs)*grid_offset
    if size(latvecs) == (2,2) && length(grid_offset) == 2
        (a,c) = diag(H)
        grid = reduce(hcat,[gridvecs*[i,j] + offset for
            (i,j)=product(0:a-1,0:c-1)])
    elseif size(latvecs) == (3,3) && length(grid_offset) == 3
        (a,c,f) = diag(H)
        grid = reduce(hcat,[gridvecs*[i,j,k] + offset for
            (i,j,k)=product(0:a-1,0:c-1,0:f-1)])
    else
        throw(ArgumentError("The lattice vectors and offset are incompatible."))
    end
    
    mapto_unitcell(grid,latvecs,inv_latvecs,"Cartesian",rtol=rtol,atol=atol)
end

@doc """
    rectangular_method(epm,N,num_cores=1;partial=true,rtol,atol)

Calculate the fermi level and band energy of an EPM with the rectangular method.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `N::Union{Integer,AbstractMatrix{<:Integer}}`: an integer, square array that relates
    the reciprocal lattice vectors `R` to the grid generating vectors `K`: `R=KN`.
    If a integer is provided, the matrix is a diagonal matrix, and each element along
    the diagonal are the integer.
- `num_cores::Integer`: the number of cores used when calculating the band energy
    in parallel.
- `rtol::Real=sqrt(eps(float(maximum(real_latvecs))))` a relative tolerance for
    floating point comparisons.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `num_unique::Integer`: the number of symmetrically unique points in the grid.
- `fermilevel::Real`: the Fermi level.
- `bandenergy::Real`: the band energy.

# Examples
```jldoctest
import Pebsi.EPMs: m11
import Pebsi.RectangularMethod: rectangular_method
rectangular_method(m11,3)
# output
(3, 0.6898935531209884, 2.021267761566246)
```
"""
function rectangular_method(epm::Union{epm₋model2D,epm₋model},
    N::Union{Integer,AbstractMatrix{<:Integer}},num_cores::Integer=1;
    partial::Bool=true,
    rtol::Real=sqrt(eps(float(maximum(epm.recip_latvecs)))), 
    atol::Real=def_atol,func=nothing)::Tuple{Integer,Real,Real}
    grid_offset = sym_offset[epm.rlat_type]
    if typeof(N) <: Integer
        if size(epm.recip_latvecs,1) == 3
            N = [N 0 0; 0 N 0; 0 0 N]
        else
            N = [N 0; 0 N]
        end
    end

    (kpoint_weights,unique_kpoints,orbits) = symreduce_grid(epm.recip_latvecs,N,
        grid_offset,epm.pointgroup,rtol=rtol,atol=atol)
    
    inv_latvecs = inv(epm.recip_latvecs)
    unique_kpoints = mapto_bz(unique_kpoints, epm.recip_latvecs, inv_latvecs,
        epm.coordinates)

    if num_cores == 1
        if func === nothing
            eigenvalues = eval_epm(unique_kpoints,epm,rtol=rtol,atol=atol)
        else
            eigenvalues = func(unique_kpoints)
        end
    else
        if func === nothing
            eigenvalues = reduce(hcat,pmap(x->eval_epm(x,epm,rtol=rtol,atol=atol,sheets=epm.sheets),
                [unique_kpoints[:,i] for i=1:size(unique_kpoints,2)]))
        else
            eigenvalues = reduce(hcat,pmap(x->func(x),[unique_kpoints[:,i] for i=1:size(unique_kpoints,2)]))
        end
    end
    
    num_unique = size(unique_kpoints,2)
    num_kpoints = sum(kpoint_weights)
    # pos = partially occupied state
    if partial
        (maxoccupied_state,pos) = divrem(epm.electrons*num_kpoints/2,1)
        if pos ≈ 1/2
            maxoccupied_state += 1
        end
    else
        maxoccupied_state = ceil(Int,round(epm.electrons*num_kpoints/2,sigdigits=12))
    end
    
    rectangle_size = abs(det(epm.recip_latvecs))/num_kpoints
    eigenweights = zeros(epm.sheets,num_unique)
    for i=1:num_unique
        eigenweights[:,i] .= kpoint_weights[i]
    end

    eigenvalues = [eigenvalues...]
    eigenweights = [eigenweights...]

    order = sortperm(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenweights = eigenweights[order]
    
    totalstates = epm.sheets*num_kpoints
    maxoccupied_state = Int(maxoccupied_state)

    counter = maxoccupied_state
    index = 0
    for i=1:totalstates
        counter -= eigenweights[i]
        if counter <= 0
            index = i
            break
        end
    end

    fermilevel = eigenvalues[index]
    bandenergy = rectangle_size*(dot(eigenweights[1:index],eigenvalues[1:index])
        + counter*eigenvalues[index])
    
    if partial
        bandenergy -= pos*rectangle_size*eigenvalues[index]
    end
    (num_unique,fermilevel,2*bandenergy)
end

@doc """
    rectangular_method(recip_latvecs,rules,electrons,cutoff,sheets,N,
        grid_offset,energy_factor;rtol,atol)

Calculate the Fermi level and band energy with the rectangular method without symmetry.

# Arguments
- `recip_latvecs::AbstractMatrix{<:Real}`: the reciprocal lattice vectors.
- `rules`: pairs where the first values are distances between lattice points and
    the second values are the pseudopotential form factors.
- `electrons::Integer`:  the number of electrons.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets::Int`: the number of sheets to include.
- `N::AbstractMatrix{<:Real}`: a square matrix relating the lattice vectors to the 
    grid generating vectors.
- `grid_offset::AbstractVector{<:Real}=[0,0]`: the offset of the grid in terms of the
    grid generating vectors. 
- `energy_factor::Real=RytoeV`: an energy conversion factor.
- `rtol::Real=sqrt(eps(float(maximum(recip_latvecs))))`: a relative tolerance.
- `atol::Real=def_atol`: an absolute tolerance.
- `func::Union{Nothing,Function}=nothing`: a k-point independent band structure model. 

# Returns
- `num_kpoints::Int64`: the number of symmetrically-unique *k*-points.
- `::Float64`:the Fermi level.
- `::Float64`: the band energy.

# Examples
```jldoctest
using Pebsi.EPMs: m11
using Pebsi.RectangularMethod: rectangular_method;
sheets = 7
N = [10 0; 0 10]
grid_offset = [0.5,0.5]
energy_conv = 1
(npts,fl,be) = rectangular_method(m11.recip_latvecs, m11.rules, m11.electrons, m11.cutoff, sheets,
    N, grid_offset, energy_conv)
abs(be - m11.bandenergy) < 1e-2
# output
true
```
"""
function rectangular_method(recip_latvecs::AbstractMatrix{<:Real},
    rules, electrons::Integer, cutoff::Real,
    sheets::Int, N::AbstractMatrix{<:Real},
    grid_offset::AbstractVector{<:Real}=[0,0], energy_factor::Real=RytoeV;
    rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),
    atol::Real=def_atol,
    func::Union{Nothing,Function}=nothing)::Tuple{Int64,Float64,Float64}

    integration_points = sample_unitcell(recip_latvecs, N, grid_offset,rtol=rtol,
        atol=atol)
    num_kpoints = size(integration_points,2)
    num_states = num_kpoints*sheets
    eigenvalues = zeros(num_states)

    for i=1:num_kpoints
        if func === nothing
        eigenvalues[1+(i-1)*sheets:(sheets*i)] = eval_epm(
            integration_points[:,i],recip_latvecs,rules,cutoff,sheets,
            energy_factor,rtol=rtol,atol=atol)
        else
            eigenvalues[1+(i-1)*sheets:(sheets*i)] = eval_epm(
                func,integration_points[:,i],sheets)    
        end
    end

    # Account for partially occupied state
    if mod(electrons*num_kpoints,2) == 1
        part = true
    else
        part = false
    end

    sort!(eigenvalues)
    occupied_states = 1:ceil(Int,electrons*num_kpoints/2)
    rectangle_size = abs(det(recip_latvecs))/num_kpoints
    fermi_level = eigenvalues[occupied_states[end]]
    band_energy = rectangle_size*sum(eigenvalues[occupied_states])

    if part
        bandenergy -= rectangle_size*eigenvalues[occupied_states[2]]
    end

    (num_kpoints,fermi_level,2*band_energy)
end

@doc """
    symreduce_grid(recip_latvecs,N,grid_offset,pointgroup;rtol,atol)

Calculate the symmetrically unique points and their weights in a GR grid.

# Arguments
- `recip_latvecs::AbstractMatrix{<:Real}`:
- `N::AbstractMatrix{<:Integer}`: the integer matrix that relates the reciprocal
    lattice vectors (R) to the grid generating vectors (K): `K = R*N`.
- `grid_offset::AbstractVector{<:Real}`: the offset of the grid in grid
    coordinates or fractions of the grid generating vectors (K). The offset in
    Cartesian coordinates is `K*gridoffset`.
- `pointgroup::AbstractVector`: the operators in the point group in matrix
    representation. They operate on points in Cartesian coordinates.
- `rtol::Real=sqrt(eps(float(maximum(recip_latvecs))))`: relative tolerance for
    floating point comparisons. This is needed when mapping points into the
    first unit cell.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons. Also
    used for mapping points to the first unit cell.

# Returns
- `kpoint_weights::Vector{<:Integer}`: a list weights or sizes of orbits for the
    symmetrically distinct points in the grid.
- `unique_kpoints::Matrix{<:Real}`: the representative points for each orbit
    as columns of a 2D array in Cartesian coordinates.
- `orbits::Vector{Matrix{<:Real}}`(optional): a list of points in each orbit.

# Examples
```jldoctest
using Pebsi.RectangularMethod: symreduce_grid
recip_latvecs = [1 0; 0 1]
N = [2 0; 0 2]
grid_offset = [0.5, 0.5]
pointgroup = [[0.0 -1.0; -1.0 0.0], [0.0 -1.0; 1.0 0.0], [-1.0 0.0; 0.0 -1.0],
    [1.0 0.0; 0.0 -1.0], [-1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0],
    [0.0 1.0; -1.0 0.0], [0.0 1.0; 1.0 0.0]]
symreduce_grid(recip_latvecs,N,grid_offset,pointgroup)
# output
([4], [0.25; 0.25], [[0.25 0.75 0.75 0.25; 0.25 0.75 0.25 0.75]])
```
"""
function symreduce_grid(recip_latvecs::AbstractMatrix{<:Real},
    N::AbstractMatrix{<:Integer}, grid_offset::AbstractVector{<:Real},
    pointgroup::AbstractVector; rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),
    atol::Real=def_atol)

    dim = length(grid_offset)
    origin = zeros(Int,dim)
    inv_rlatvecs = inv(recip_latvecs)
    N = matrix(ZZ,N)

    (H,U) = hnf_with_transform(N)
    (D,A,B) = snf_with_transform(H)
    (D,A,N,H) = [Array(x) for x=[D,A,N,H]]
    (D,A,N,H) = [convert(Array{Int64,2},x) for x=[D,A,N,H]]
    d = diag(D)
    K = inv(H)*recip_latvecs
    invK = inv_rlatvecs*H

    offset = mapto_unitcell(K*grid_offset,recip_latvecs,inv(recip_latvecs),
        "Cartesian",rtol=rtol,atol=atol)
    grid = sample_unitcell(recip_latvecs,N,grid_offset,rtol=rtol,atol=atol)
    num_kpoints = size(grid,2)
    unique_count = 0

    # hash_table keeps track of k-points within orbits
    hash_table = zeros(Int,num_kpoints)
    # The representative k-point for each orbit
    first_kpoint = zeros(Int,num_kpoints)
    kpoint_weights = zeros(Int,num_kpoints)
    # A list of all indices, used as a failsafe
    indices = zeros(Int,num_kpoints)

    orbits = [zeros(Int,length(pointgroup)) for i=1:num_kpoints]
    keep = zeros(Int,num_kpoints)
    for i=1:num_kpoints
        orbsize = 0
        # Remove the offset because indexing doesn't work with shifted grids.
        indexᵢ = kpoint_index(grid[:,i],offset,invK,A,d)
        indices[i] = indexᵢ
        # Move on if this k-point is already part of an orbit.
        if hash_table[indexᵢ] != 0
            continue
        end
        unique_count += 1
        hash_table[indexᵢ] = unique_count
        first_kpoint[indexᵢ] = i
        kpoint_weights[indexᵢ] = 1
        orbits[i][1] = indexᵢ
        orbsize += 1
        keep[i] = 1
        for (j,op) in enumerate(pointgroup)
            rot_point = mapto_unitcell(op*grid[:,i],recip_latvecs,inv_rlatvecs,
                "Cartesian",rtol=rtol,atol=atol)
            test = mapto_unitcell(rot_point-offset,K,invK,"Cartesian",
                rtol=rtol,atol=atol)
            if !(isapprox(test,origin,rtol=rtol,atol=atol))
                continue
            end

            indexⱼ = kpoint_index(rot_point,offset,invK,A,d)
            if hash_table[indexⱼ] == 0
                hash_table[indexⱼ] = unique_count
                kpoint_weights[indexᵢ] += 1
                orbsize += 1
                orbits[i][orbsize] = indexⱼ
            end
        end
        orbits[i] = orbits[i][1:orbsize]
    end
    orbits = orbits[findall(x->x==1,keep)]
    if sort(indices) != 1:num_kpoints
        error("The k-point indices are calculated incorrectly.")
    end

    nonzero_orbits = findall(x->x!=0,kpoint_weights)
    unique_kpoints = grid[:,first_kpoint[nonzero_orbits]]
    kpoint_weights = kpoint_weights[nonzero_orbits]

    order = sortperm(indices)
    orbits = [grid[:,order[orb]] for orb = orbits]
    (kpoint_weights, unique_kpoints, orbits)
end

@doc """
    convert_mixedradix(dials,dial_sizes)

Convert a mixed-radix number to an integer.

# Arguments
- `dial::AbstractVector{<:Integer}`: the mixed radix number as integers
    in a 1D array.
- `dial_sizes::AbstractVector{<:Integer}`: the maximum value of each dial
    as integers in a 1D array.

# Returns
    `val::Integer`: the mixed radix-number as an integer, such as the position
    in a 1D array, for example.

# Examples
```jldoctest
using Pebsi.RectangularMethod: convert_mixedradix
dials = [1,2]
dial_sizes = [3,4]
convert_mixedradix(dials,dial_sizes)
# output
8
```
"""
function convert_mixedradix(dial::AbstractVector{<:Integer}, 
    dial_sizes::AbstractVector{<:Integer})::Integer
    val = 1 + dial[1]
    for (i,x)=enumerate(dial[2:end])
        val += x*prod(dial_sizes[1:i])
    end
    val
end

@doc """
    kpoint_index(point,offset,invK,A,snf_diag)

Calculate the index of a point in a generalized regular grid.

# Arguments
- `point::AbstractVector{<:Real}`: a point in the grid in Cartesian
    coordinates.
- `offset::AbstractVector{<:Real}`: the offset of the grid in Cartesian
    coordinates.
- `invK::AbstractMatrix{<:Real}`: the inverse of the array with the grid
    generating vectors as columns.
- `A::AbstractMatrix{<:Real}`: the left transform for converting an integer
    array into Smith normal form: `N = A*S*B` where N in an integer array.
- `snf_diag::AbstractVector{<:Real}`: the diagonal elements of the integer
    array in Smith normal form.

# Returns
- `::Integer`: the index of the point.

# Examples
```jldoctest
using Pebsi.RectangularMethod: kpoint_index
point = [0.5, 0.5]
offset = [0,0]
invK = [0.25  0; 0 0.25]
A = [1 1; -1 0]
snf_diag = [4,4]
kpoint_index(point,offset,invK,A,snf_diag)
# output
1
```
"""
function kpoint_index(point::AbstractVector{<:Real},
        offset::AbstractVector{<:Real}, invK::AbstractMatrix{<:Real},
        A::AbstractMatrix{<:Real}, snf_diag::AbstractVector{<:Real})::Integer
    dial = mod.(A*round.(Int,invK*(point-offset)),snf_diag)
    convert_mixedradix(dial,snf_diag)
end

@doc """
    kpoint_index(points,offset,invK,A,snf_diag)

Calculate the indices of points in an array as columns.
"""
function kpoint_index(points::AbstractMatrix{<:Real},
        offset::AbstractVector{<:Real}, invK::AbstractMatrix{<:Real},
        A::AbstractMatrix{<:Real},
        snf_diag::AbstractVector{<:Real})::AbstractArray{Integer,1}

    [kpoint_index(points[:,i],offset,invK,A,snf_diag) for i=1:size(points,2)]
end

@doc """
    calculate_orbits(grid,pointgroup,latvecs;rtol,atol)

Calculate the points of the grid in each orbit the hard way.

# Arguments
- `grid::AbstractMatrix{<:Real}`: the points in the grid as columns of an
    array in Cartesian coordinates.
- `pointgroup::AbstractVector`: the operators in the point group in nested
    array. They operator on points in Cartesian coordinates.
- `latvecs::AbstractMatrix{<:Real}`: the lattice vectors as columns of an
    array.
- `rtol::Real=sqrt(eps(float(maximum(grid))))`: relative tolerance.
- `atol::Real=def_atol`: absolute tolerance.

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
1-element Vector{Matrix{Float64}}:
 [0.75 0.75 0.25 0.25; 0.75 0.25 0.75 0.25]
```
"""
function calculate_orbits(grid::AbstractMatrix{<:Real},
    pointgroup,latvecs::AbstractMatrix{<:Real};
    rtol::Real=sqrt(eps(float(maximum(grid)))),atol::Real=def_atol)::AbstractArray

    inv_latvecs = inv(latvecs)
    coordinates = "Cartesian"
    dim = size(grid,1)
    uc_grid = mapto_unitcell(grid,latvecs,inv_latvecs,coordinates,
        rtol=rtol,atol=atol)
    numpts = 0
    orbits = [zeros(dim,length(pointgroup)) for i=1:size(uc_grid,2)]
    orbit = zeros(dim,length(pointgroup))
    gridcopy =  uc_grid
    while size(gridcopy) != (dim,0)
        pt = gridcopy[:,1]
        numpts += 1
        orbit = zeros(dim,length(pointgroup))
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
                if size(gridcopy) == (dim,0)
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
