module RectangularMethod

import SymmetryReduceBZ.Symmetry: mapto_unitcell, make_primitive,
    calc_spacegroup
import SymmetryReduceBZ.Lattices: get_recip_latvecs
import Base.Iterators: product
import LinearAlgebra: det, diag, dot
import AbstractAlgebra: ZZ, matrix, snf_with_transform, hnf_with_transform, hnf

include("EPMs.jl")
import .EPMs: eval_epm,RytoeV,eVtoRy


@doc """
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

    mapto_unitcell(grid,latvecs,inv_latvecs,"Cartesian",rtol,atol)
end

@doc """
    rectangular_method(real_latvecs,atom_types,atom_pos,rules,electrons,cutoff,
        sheets,N,grid_offset,convention,coordinates,energy_factor,rtol,atol)

# Arguments
- `real_latvecs::AbstractArray{<:Real,2}`: the basis of the lattice as columns
    of an array.
- `atom_types::AbstractArray{<:Int,1}`: a list of atom types as integers.
- `atom_pos::AbstractArray{<:Real,2}`: the positions of atoms in the crystal
    structure as columns of an array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `electrons::Integer`: the number of free electrons in the unit cell.
- `cutoff::Real`: the Fourier expansion cutoff.
- `sheets::UnitRange{<:Int}`: the sheets of the band structure included in the
    calculation. This must begin with 1 for the result to make any sense.
- `N::AbstractArray{<:Integer,2}`: an integer, square array that relates the
    reciprocal lattice vectors `R` to the grid generating vectors `K`: `R=KN`.
- `grid_offset::AbstractArray{<:Real,1}=[0,0]`: the offset of the grid in grid
    coordinates (fractions of the grid generating vectors).
- `convention::String="ordinary"`: the convention used to go between real and
    reciprocal space. The two conventions are ordinary (temporal) frequency and
    angular frequency. The transformation from real to reciprocal space is
    unitary if the convention is ordinary.
- `coordinates::String`: indicates the positions of the atoms are in \"lattice\"
    or \"Cartesian\" coordinates.
- `energy_conversion_factor::Real=RytoeV`: converts the energy eigenvalue units
    from the energy unit used for `rules` to an alternative energy unit.
- `rtol::Real=sqrt(eps(float(maximum(real_latvecs))))` a relative tolerance for
    floating point comparisons.
- `atol::Real=0.0`: an absolute tolerance for floating point comparisons.

# Returns
- `num_unique::Integer`: the number of symmetrically unique points in the grid.
- `fermilevel::Real`: the Fermi level.
- `bandenergy::Real`: the band energy.

# Examples
```jldoctest
import Pebsi.RectangularMethod: rectangular_method

real_latvecs = [1 0; 0 1]
atom_types = [0]
atom_pos = Array([0 0]')
coordinates = "Cartesian"
rules = Dict(1.00 => -0.23, 1.41 => 0.12)
electrons = 6
cutoff = 6.1
sheets = 1:10
N = [10 0; 0 10]
grid_offset = [0.5,0.5]
convention = "ordinary"
coordinates = "Cartesian"
energy_factor = 1
rectangular_method(real_latvecs,atom_types,atom_pos,rules,electrons,cutoff,
    sheets,N,grid_offset,convention,coordinates,energy_factor)
# output
(38, 0.8913900782229439, 1.0409313912201126)
```
"""
function rectangular_method(real_latvecs::AbstractArray{<:Real,2},
    atom_types::AbstractArray{<:Integer,1}, atom_pos::AbstractArray{<:Real,2},
    rules::Dict{Float64,Float64}, electrons::Integer, cutoff::Real,
    sheets::UnitRange{<:Int}, N::AbstractArray{<:Integer,2},
    grid_offset::AbstractArray{<:Real,1}=[0,0], convention::String="ordinary",
    coordinates::String="Cartesian", energy_factor::Real=RytoeV,
    rtol::Real=sqrt(eps(float(maximum(real_latvecs)))),
    atol::Real=0.0)::Tuple{Int64,Float64,Float64}

    (atom_types,atom_pos,real_latvecs)=make_primitive(real_latvecs,atom_types,
        atom_pos,coordinates,rtol,atol)
    (frac_trans,pointgroup) = calc_spacegroup(real_latvecs,atom_types,atom_pos,
        coordinates,rtol,atol)

    recip_latvecs = get_recip_latvecs(real_latvecs,convention)
    (kpoint_weights,unique_kpoints,orbits) = symreduce_grid(recip_latvecs,N,
        grid_offset,pointgroup,rtol,atol)
    eigenvalues = mapslices(x->eval_epm(x,recip_latvecs,rules,cutoff,sheets,
        energy_factor,rtol,atol),unique_kpoints,dims=1)

    num_unique = size(unique_kpoints,2)
    num_kpoints = sum(kpoint_weights)

    maxoccupied_state = ceil(Int,round(electrons*num_kpoints/2,sigdigits=12))
    rectangle_size = abs(det(recip_latvecs))/num_kpoints

    eigenweights = zeros(sheets[end],num_unique)
    for i=1:num_unique
        eigenweights[:,i] .= kpoint_weights[i]
    end

    eigenvalues = [eigenvalues...]
    eigenweights = [eigenweights...]

    order = sortperm(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenweights = eigenweights[order]

    totalstates = sheets[end]*num_kpoints
    counter = maxoccupied_state
    index = 0
    for i=1:num_kpoints*sheets[end]
        counter -= eigenweights[i]
        if counter <= 0
            index = i
            break
        end
    end

    fermilevel = eigenvalues[index]
    bandenergy = rectangle_size*dot(eigenweights[1:index],eigenvalues[1:index])

    (num_unique,fermilevel,bandenergy)
end

@doc """
    rectangular_method(recip_latvecs,rules,electrons,cutoff,sheets,N,
        grid_offset,energy_factor,rtol,atol)

Calculate the Fermi level and band energy with the rectangular method without symmetry.
"""
function rectangular_method(recip_latvecs::AbstractArray{<:Real,2},
    rules::Dict{Float64,Float64}, electrons::Integer, cutoff::Real,
    sheets::UnitRange{<:Int}, N::AbstractArray{<:Real,2},
    grid_offset::AbstractArray{<:Real,1}=[0,0], energy_factor::Real=RytoeV,
    rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),
    atol::Real=0.0)::Tuple{Int64,Float64,Float64}

    integration_points = sample_unitcell(recip_latvecs, N, grid_offset,rtol,
        atol)
    num_kpoints = size(integration_points,2)
    num_states = num_kpoints*sheets[end]
    eigenvalues = zeros(num_states)

    for i=1:num_kpoints
        eigenvalues[1+(i-1)*sheets[end]:(sheets[end]*i)] = eval_epm(
            integration_points[:,i],recip_latvecs,rules,cutoff,sheets,
            energy_factor,rtol,atol)
    end

    sort!(eigenvalues)
    occupied_states = 1:ceil(Int,electrons*num_kpoints/2)
    rectangle_size = det(recip_latvecs)/num_kpoints
    fermi_level = eigenvalues[occupied_states[end]]
    band_energy = rectangle_size*sum(eigenvalues[occupied_states])

    (num_kpoints,fermi_level,band_energy)
end

@doc """
    symreduce_grid(recip_latvecs,N,grid_offset,pointgroup,rtol,atol)

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
    pointgroup::AbstractArray,
    rtol::Real=sqrt(eps(float(maximum(recip_latvecs)))),
    atol::Real=1e-9)

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
        "Cartesian",rtol,atol)
    grid = sample_unitcell(recip_latvecs,N,grid_offset,rtol,atol)
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
                "Cartesian",rtol,atol)
            test = mapto_unitcell(rot_point-offset,K,invK,"Cartesian",rtol,atol)
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
- `dial::AbstractArray{<:Integer,1}`: the mixed radix number as integers
    in a 1D array.
- `dial_sizes::AbstractArray{<:Integer,1}`: the maximum value of each dial
    as integers in a 1D array.

# Returns
    `val::Integer`: the mixed radix-number as an integer, such as the position
    in a 1D array, for example.

# Examples
```jldoctest
dials = [1,2]
dial_sizes = [3,4]
convert_mixedradix(dials,dial_sizes)
# output
8
```
"""
function convert_mixedradix(dial::AbstractArray{<:Integer,1},
        dial_sizes::AbstractArray{<:Integer,1})::Integer
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
- `point::AbstractArray{<:Real,1}`: a point in the grid in Cartesian
    coordinates.
- `offset::AbstractArray{<:Real,1}`: the offset of the grid in Cartesian
    coordinates.
- `invK::AbstractArray{<:Real,2}`: the inverse of the array with the grid
    generating vectors as columns.
- `A::AbstractArray{<:Real,2}`: the left transform for converting an integer
    array into Smith normal form: `N = A*S*B` where N in an integer array.
- `snf_diag::AbstractArray{<:Real,1}`: the diagonal elements of the integer
    array in Smith normal form.

# Returns
- `::Integer`: the index of the point.

# Examples
```jldoctest
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
function kpoint_index(point::AbstractArray{<:Real,1},
        offset::AbstractArray{<:Real,1}, invK::AbstractArray{<:Real,2},
        A::AbstractArray{<:Real,2}, snf_diag::AbstractArray{<:Real,1})::Integer
    dial = mod.(A*round.(Int,invK*(point-offset)),snf_diag)
    convert_mixedradix(dial,snf_diag)
end

@doc """
    kpoint_index(points,offset,invK,A,snf_diag)

Calculate the indices of points in an array as columns.
"""
function kpoint_index(points::AbstractArray{<:Real,2},
        offset::AbstractArray{<:Real,1}, invK::AbstractArray{<:Real,2},
        A::AbstractArray{<:Real,2},
        snf_diag::AbstractArray{<:Real,1})::AbstractArray{Integer,1}

    [kpoint_index(points[:,i],offset,invK,A,snf_diag) for i=1:size(points,2)]
end

@doc """
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
import Pebsi.RectangularMethod: calculate_orbits, coordinatescell
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
    dim = size(grid,1)
    uc_grid = mapto_unitcell(grid,latvecs,inv_latvecs,coordinates,rtol,atol)
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
