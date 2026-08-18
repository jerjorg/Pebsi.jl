@doc """
    refine_mesh!(epm,ebs)

Perform one iteration of adaptive refinement. See the composite type
`BandStructure` for refinement options.

# Arguments
- `epm::Union{EPM2D,EPM}`: an empirical pseudopotential.
- `ebs::BandStructure`: a quadratic approximation of the band structure.

# Returns
- `ebs::BandStructure`: updated interval coefficients, Bezier coefficients, mesh, 
    extended mesh, Fermi level, band energy, ... for the quadratic approximation.

# Examples
```jldoctest
using Suppressor
using Pebsi.EPMs: m21
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!, refine_mesh!
epm = m21
ebs = init_bandstructure(epm);
@suppress calc_flbe!(epm,ebs)
@suppress refine_mesh!(epm,ebs)
abs(ebs.bandenergy - epm.bandenergy) < 1e-2
# output
true
```
"""
function refine_mesh!(epm::Union{EPM2D,EPM},ebs::BandStructure)
    spatial = pyimport("scipy.spatial")
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    err_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.target_accuracy
    faerr_cutoff = [simplex_size(s)/epm.ibz.volume for s=simplices]*ebs.fermiarea_eps
     
    n = def_min_split
    # Refine the tiles with the most error
    if ebs.refine_method == refine_most_error
        splitpos = sortperm(abs.(ebs.bandenergy_errors),rev=true)
        if length(splitpos) > n splitpos = splitpos[1:n] end

    # Refine the tiles with too much error (given the tiles' sizes).
    elseif ebs.refine_method == refine_above_allowed
        splitpos = filter(x -> x > 0,[abs(ebs.bandenergy_errors[i]) > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])

    # Refine a fraction of the number of tiles that have too much error.
    elseif ebs.refine_method == refine_fraction_above_allowed
        splitpos = filter(x -> x>0,[abs(ebs.bandenergy_errors[i]) > err_cutoff[i] ? i : 0 for i=1:length(err_cutoff)])
        if length(splitpos) > n
            order = sortperm(abs.(ebs.bandenergy_errors[splitpos]),rev=true)
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        end 
    # Refine where the Fermi area errors are too much. Scale Fermi area errors
    # by Fermi level to get errors in terms of band energy
    elseif ebs.refine_method == refine_fermiarea_above_allowed
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
    elseif ebs.refine_method == refine_fermiarea_largest
        if sum(ebs.fermiarea_errors) < ebs.fermiarea_eps
            println("Switching to band energy refinement.")
            ebs.refine_method = 6
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
    elseif ebs.refine_method == refine_partially_occupied
        # Only split triangles that are partially occupied
        splitpos = sort(unique([any(x->x==1,po) ? i : 0 for (i,po) in 
            enumerate(ebs.partially_occupied)]))[2:end]
        order = sortperm(abs.(ebs.bandenergy_errors[splitpos]),rev=true)
        if length(splitpos) > n
            splitpos = splitpos[order[1:round(Int,length(order)*def_frac_refined)]]
        else
            splitpos = splitpos[order]
        end
    elseif ebs.refine_method == refine_largest_fraction
        order = sortperm(abs.(ebs.bandenergy_errors),rev=true)
        if length(order) > n/def_frac_refined
            numsplit = round(Int,length(order)*def_frac_refined)
            splitpos = order[1:numsplit]
        elseif length(order) > n
            splitpos = order[1:n]
        else
            splitpos = order
        end        
    else
        ArgumentError("Unhandled RefineMethod: $(ebs.refine_method)")
    end

    # Split fewer simplices if too many k-points are added.
    if ebs.stop_criterion == stop_kpoint_target
        dim = size(epm.recip_latvecs,1)
        p = if dim == 2 3 else 6 end
        nkpts = size(ebs.eigenvals,2) - 2^dim
        max_added = p*length(splitpos)
        tot_kpts = nkpts + max_added
        if ebs.target_kpoints < tot_kpts
            numremove = round(Integer,(tot_kpts - ebs.target_kpoints)/p)
            if numremove >= length(splitpos) numremove = length(splitpos) - 1 end
            splitpos = splitpos[1:end-numremove]
        end
    end

    println("Number of split simplices: ", length(splitpos))
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
    if ebs.sample_method == sample_center
        new_meshpts = reduce(hcat,[barytocart(centerpt,s) for s=simplices[splitpos]])
    # Point at the midpoints of all edges of the triangle
    elseif ebs.sample_method == sample_edge_midpoints
        new_meshpts = reduce(hcat,[barytocart(edgepts,s) for s=simplices[splitpos]])
    # If the error is 2x greater than the tolerance, split edges. Otherwise,
    # sample at the center of the triangle.
    elseif ebs.sample_method == sample_adaptive
        sample_type = [
            abs(ebs.bandenergy_errors[i]) > def_allowed_err_ratio*err_cutoff[i] ? 2 : 1 for i=splitpos]
        new_meshpts = reduce(hcat,[sample_type[i] == 1 ? 
        barytocart(centerpt,simplices[splitpos[i]]) :
        barytocart(edgepts,simplices[splitpos[i]])
        for i=1:length(splitpos)])
    else
        ArgumentError("Unhandled SampleMethod: $(ebs.sample_method)")
    end

    # Remove duplicates from the new mesh points.
    new_meshpts = unique_points(new_meshpts,rtol=ebs.rtol,atol=ebs.atol)
    new_eigvals = eval_epm(new_meshpts,epm,rtol=ebs.rtol,atol=ebs.atol)
    ebs.eigenvals = [ebs.eigenvals new_eigvals]

    println("Unique points added: $(size(new_meshpts,2))")

    # There should technically be an additional step at this point where
    # symmetrically equivalent points are removed from `new_meshpts` (points
    # on different boundaries of the IBZ may be rotationally or translationally
    # equivalent, but I figure the chances of two points being equivalent are
    # pretty slim and the extra cost isn't too great, but I could be wrong, so 
    # I'm making a note.

    cv_pointsᵢ = get_cvpts(Matrix(ebs.mesh.points'),epm.ibz,atol=ebs.atol)
    # Calculate the maximum distance between neighboring points
    bound_limit = def_max_neighbor_tol*maximum(
        reduce(vcat,[[norm(ebs.mesh.points[i,:] - ebs.mesh.points[j,:]) 
                    for j=get_neighbors(i,ebs.mesh,ebs.num_near_neigh)] for i=cv_pointsᵢ]))

    borders,distfun = ibz_borders(epm.ibz)
    bztrans = bz_translations(dim)
     
    # The number of points in the mesh before adding new points.
    s = size(ebs.mesh.points,1)
    m = maximum(ebs.sym_unique)

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
        ebs.sym_unique = [ebs.sym_unique[1:m]; new_ind; sym_ind[1:nₑ]] 
        ebs.ext_mesh = spatial.Delaunay([ebs.ext_mesh.points[1:m,:]; new_meshpts'; neighbors[:,1:nₑ]'])
    else
        ebs.sym_unique = [ebs.sym_unique[1:m]; new_ind; sym_ind[1:nₘ]; ebs.sym_unique[m+1:end];
            sym_ind[nₘ+1:nₑ]]
        ebs.ext_mesh = spatial.Delaunay([ebs.ext_mesh.points[1:m,:]; new_meshpts'; neighbors[:,1:nₘ]';
            ebs.ext_mesh.points[m+1:end,:]; neighbors[:,nₘ+1:nₑ]']) 
    end

    ebs.simplicesᵢ = notbox_simplices(ebs.mesh)
    coeffs = [get_intercoeffs(index,mesh=ebs.mesh,ext_mesh=ebs.ext_mesh,
    sym_unique=ebs.sym_unique,eigenvals=ebs.eigenvals,simplicesᵢ=ebs.simplicesᵢ,
    degree=ebs.polydegree,fatten=ebs.fatten,num_near_neigh=ebs.num_near_neigh,
    neighbor_method=ebs.neighbor_method,epm=epm) for index=1:length(ebs.simplicesᵢ)]

    ebs.mesh_intcoeffs = [coeffs[index][1] for index=1:length(ebs.simplicesᵢ)]
    ebs.mesh_bezcoeffs = [coeffs[index][2] for index=1:length(ebs.simplicesᵢ)]        
    ebs
end

@doc """
    stop_refinement(ebs)

Select a condition that determines if refinement may stop.

# Arguments
- `epm::Union{EPM,EPM2D}`: a empirical pseudopotential model.
- `ebs::BandStructure`: a band structure object

# Returns
- `stop::Bool`: a boolean that tells when refinement may stop.

# Examples
```
using Pebsi.QuadraticIntegration, Pebsi.EPMs
epm = m11
ebs = init_bandstructure(epm)
calc_flbe!(epm,ebs)
stop_refinement!(ebs)
```
"""
function stop_refinement!(epm::Union{EPM,EPM2D},ebs::BandStructure,
    prevbe)::Bool
    stop = false
    if ebs.stop_criterion == stop_total_error
        stop = abs(sum(ebs.bandenergy_errors)) < ebs.target_accuracy
    elseif ebs.stop_criterion == stop_energy_change
        stop = abs(ebs.bandenergy - prevbe) < ebs.target_accuracy
    elseif ebs.stop_criterion == stop_interval
        db,da,dltol,datol = get_tolerances(epm,ebs)
        ebs.fermiarea_eps = datol
        stop = db/da*diff(ebs.fermiarea_interval)[1]/2 < ebs.target_accuracy
    elseif ebs.stop_criterion == stop_kpoint_target 
        nkpts = size(ebs.eigenvals,2) - 2^size(epm.recip_latvecs,1)
        stop = ((ebs.target_kpoints - nkpts) < def_stop_kpoint_tol*nkpts) || (nkpts >  ebs.target_kpoints)
    else
        error("Unhandled StopCriterion: $(ebs.stop_criterion)")
    end
    stop
end
