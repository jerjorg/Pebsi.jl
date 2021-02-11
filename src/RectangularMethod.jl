module RectangularMethod

import Pebsi.EPMs: eval_EPM
import Base.Iterators: product
import LinearAlgebra: det

"""
    sample_unitcell(recip_latvecs,ndivs,grid_offset)

Create a regular grid over the unit cell.
"""
function sample_unitcell(latvecs::AbstractArray{<:Real,2},
    ndivs::AbstractArray{<:Int,1}, grid_offset::AbstractArray{<:Real,1}=[0,0])

    if size(latvecs) == (2,2) && length(grid_offset) == 2
        reduce(hcat,[latvecs./ndivs*([i,j] + grid_offset) for
            (i,j)=product(0:ndivs[1]-1,0:ndivs[2]-1)])
    elseif size(latvecs) == (3,3) && length(grid_offset) == 3
        reduce(hcat,[latvecs./ndivs*([i,j,k] + grid_offset) for
            (i,j,k)=product(0:ndivs[1]-1,0:ndivs[2]-1,0:ndivs[3]-1)])
    else
        throw(ArgumentError("The lattice vectors and offset are incompatible."))
    end
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
    sym_reduce_grid(grid,recip_latvecs,ndivs,pointgroup,rtol,atol)

Calculate the symmetrically unique k-points and their weights
"""
function sym_reduce_grid(grid, recip_latvecs, ndivs, pointgroup,
        rtol=sqrt(eps(float(maximum(recip_latvecs)))),atol=0.0)

    if length(ndivs) == 2 && size(recip_latvecs) == (2,2) && size(grid,1) == 2
        dim = 2
        N = [ndivs[1] 0; 0 ndivs[2]]
    elseif length(ndivs) == 3 && size(recip_latvecs) == (3,3) && size(grid,1) == 3
        dim = 3
        N = [ndivs[1] 0 0; 0 ndivs[2] 0; 0 0 ndivs[3]]
    else
        throw(ArgumentError("Only regular grids in 2D or 3D can be reduced."))
    end

    K = inv(Array(N))*recip_latvecs
    invK = inv(K)
    N = matrix(ZZ,N)
    (D,A,B) = snf_with_transform(N)
    (D,A) = [Array(x) for x=[D,A]]
    (D,A) = [convert(Array{Int64,2},x) for x=[D,A]]

    if dim == 2
        d = [D[1,1],D[2,2]]
    else
        d = [D[1,1],D[2,2],D[3,3]]
    end

    inv_rlatvecs = inv(recip_latvecs)
    num_kpoints = size(grid,2)
    unique_count = 0

    # hash_table keeps track k-points within orbits
    hash_table = zeros(Int,num_kpoints)
    # The representative k-point for each orbit
    first_kpoint = zeros(Int,num_kpoints)
    kpoint_weights = zeros(Int,num_kpoints)
    # A list of indices, used as a failsafe
    indices = zeros(Int,num_kpoints)
    for i=1:num_kpoints
        dialᵢ = mod.(A*invK*grid[:,i],d)
        indexᵢ = round(Int64,1 + dialᵢ[1] + dialᵢ[2]*d[1])
        indices[i] = indexᵢ
        # Move on if this k-point is already part of an orbit.
        if hash_table[indexᵢ] != 0
            continue
        end
        unique_count += 1
        hash_table[indexᵢ] = unique_count
        first_kpoint[indexᵢ] = i
        kpoint_weights[indexᵢ] = 1

        for op in pointgroup
            rot_point = mapto_unitcell(op*grid[:,i],recip_latvecs,inv_rlatvecs,
                "Cartesian",rtol,atol)
            if !(isapprox(mod.(invK*rot_point,1),[0,0],rtol=rtol,atol=atol))
                continue
            end
            dialⱼ = mod.(A*invK*rot_point,d)
            indexⱼ = round(Int64,1 + dialⱼ[1] + dialⱼ[2]*d[1])
            if hash_table[indexⱼ] == 0
                hash_table[indexⱼ] = unique_count
                kpoint_weights[indexᵢ] += 1
            end
        end
    end

    if sort(indices) != collect(1:num_kpoints)
        error("The k-point indices are calculated incorrectly.")
    end

    indices = findall(x->x!=0,kpoint_weights)
    unique_kpoints = grid[:,first_kpoint[indices]]
    kpoint_weights = kpoint_weights[indices]

    (kpoint_weights, unique_kpoints)
end



end # module
