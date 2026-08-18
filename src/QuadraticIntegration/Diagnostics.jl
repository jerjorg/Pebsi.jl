@doc """
    quadratic_method(epm,stop_criterion,init_msize,num_near_neigh,num_neighbors,
        fermiarea_eps,target_accuracy,fermilevel_method,refine_method,sample_method,
        neighbor_method,fatten,rtol,atol,uniform)

Calculate the band energy using uniform or adaptive quadratic integation.

# Arguments
- `epm::Union{EPM2D,EPM}`: an empirical pseudopotential

See the documentation for `BandStructure` for descriptions of the optional arguments.

# Returns
- `ebs::BandStructure`: a quadratic approximation of the band structure.

# Examples
```
using Pebsi.EPMs: m51
using Pebsi.QuadraticIntegration: quadratic_method
epm = m51
ebs = quadratic_method(epm,target_accuracy=1e-2)
abs(ebs.bandenergy - epm.bandenergy) < 1e-1
```
"""
function quadratic_method(epm::Union{EPM2D,EPM};
    init_msize::Int=def_init_msize, num_near_neigh::Int=def_num_near_neigh,
    num_neighbors::Union{Nothing,Int}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::FermiLevelMethod=def_fermilevel_method, 
    refine_method::RefineMethod=def_refine_method,
    sample_method::SampleMethod=def_sample_method, 
    neighbor_method::NeighborMethod=def_neighbor_method,
    fatten::Real=def_fatten, rtol::Real=def_rtol, atol::Real=def_atol,
    uniform::Bool=def_uniform, weighted::Bool=def_weighted,
    constrained::Bool=def_constrained, stop_criterion::StopCriterion=def_stop_criterion,
    target_kpoints::Integer=def_target_kpoints)::BandStructure

    dim = size(epm.recip_latvecs,1)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end
    ebs = init_bandstructure(epm,init_msize=init_msize, num_near_neigh=num_near_neigh,
        num_neighbors=num_neighbors,fermiarea_eps=fermiarea_eps, 
        target_accuracy=target_accuracy, fermilevel_method=fermilevel_method, 
        refine_method=refine_method, sample_method=sample_method, 
        neighbor_method=neighbor_method, fatten=fatten, weighted=weighted, 
        constrained=constrained, stop_criterion=stop_criterion,
        target_kpoints=target_kpoints, rtol=rtol, atol=atol)
    calc_flbe!(epm,ebs)
    if uniform return ebs end
    counter = 0; prevbe = 1e9; tmp = []
    sd = 3 # rounding parameter for print statements
    stop = stop_refinement!(epm,ebs,prevbe)
    while !stop
        counter += 1; refine_mesh!(epm,ebs); calc_flbe!(epm,ebs)
        stop = stop_refinement!(epm,ebs,prevbe)
        prevbe = ebs.bandenergy 
        if counter > max_refine_steps
            @warn "Failed to calculate the band energy to within the desired accuracy $(ebs.target_accuracy) after $(max_refine_steps) iterations."
            break
        end
        diffbe = abs(ebs.bandenergy - prevbe)
        ϵᵦ = abs(ebs.bandenergy - epm.bandenergy) 
        ϵₗ = abs(ebs.fermilevel - epm.fermilevel)
        db,da,dltol,datol = get_tolerances(epm,ebs)
        println("Number of simplices: ", length(ebs.simplicesᵢ)) 
        # println("True errors")
        # println("ϵᵦ: ", round(ϵᵦ,sigdigits=sd))
        # println("ϵₗ: ", round(ϵₗ,sigdigits=sd))

        # println("Estimated band energy error")
        # println("δB/ϵᵦ: ", round(diffbe/ϵᵦ,sigdigits=sd))
        # println("ΔB/ϵᵦ: ", round(sum(ebs.bandenergy_errors)/ϵᵦ,sigdigits=sd))
        # println("dB/dL*ΔLₜ/ϵᵦ: ", round(db*diff(ebs.fermilevel_interval)[1]/2/ϵᵦ,sigdigits=sd))
        # println("dB/dA*ΔAₜ/ϵᵦ: ", round(db/da*diff(ebs.fermiarea_interval)[1]/2/ϵᵦ,sigdigits=sd))
        
        # println("Estimated Fermi level error")
        # println("ΔL/ϵₗ: ", round(diff(ebs.fermilevel_interval)[1]/2/ϵₗ,sigdigits=sd))
        # println("Estimated Fermi area errors")
        # println("ΔA: ", round(diff(ebs.fermiarea_interval)[1]/2,sigdigits=sd))
        # println("dA/dL*ΔLₜ: ", round(da*diff(ebs.fermilevel_interval)[1]/2,sigdigits=sd))
        
        # println("Derivatives and tolerances")
        # println("dB/dL dA/dL dB/dA ΔAₜ ΔLₜ = ",round.([db,da,db/da,dltol,datol],sigdigits=sd),"\n")
    end
    ebs
end

"""
    truebe(epm,ebs,ndivs,num_cores,triangles)
    
Calculate (roughly) the true band energy error for each quadratic triangle.

# Arguments
- `epm::BandStructure`: a quadratic approximation of the band structure
- `ebs::EPM2D`: an empirical pseudopotential
- `ndivs::Integer`: the number of divisions of triangles when computing the band
    energy component within each triangle using the rectangular method with a triangular
    base.
- `num_cores::Integer=1`: the number of cores to use when computing in parallel.
- `triangles`: a list of triangles over which to compute the "true" band energy. If
    nothing is provided, compute over all triangles in the approximation.

# Returns
- `sigma_be::Vector{Real}`: the true sigma band energy over each triangle.
- `part_be::Vector{Real}`: the true partial band energy over each triangle.

# Examples
```
using Pebsi.EPMs: m51
using Pebsi.QuadraticIntegration: quadratic_method, init_bandstructure, calc_flbe!, truebe
epm = m51
ebs = init_bandstructure(epm)
ebs = calc_flbe!(epm,ebs)
sigma_be,part_be = truebe(epm,ebs,10)
```
"""
function truebe(epm::EPM2D,ebs::BandStructure,ndivs::Integer;
    num_cores::Integer=1,triangles::Union{Nothing,Integer}=nothing)
    dim = 2
    deg = 2
    num_tri = length(ebs.simplicesᵢ)
    num_sheets = epm.sheets
    
    if triangles == nothing
        triangles = 1:num_tri
    end

    # Locate the highest occupied sheet
    max_sheet = 0
    for tri = triangles
        for sheet=1:num_sheets
            if ebs.partially_occupied[tri][sheet] == 1
                if sheet > max_sheet max_sheet = sheet end
            end
        end
    end
    max_sheet += 2

    # Identify occupied, partially occupied, and unoccupied sheets
    bpts = trimesh(ndivs)
    vals = zeros(max_sheet,size(bpts,2))
    pts = zeros(3,length(ebs.simplicesᵢ))
    sigma_be = [0. for _=1:length(triangles)]
    part_be = [0. for _=1:length(triangles)]
    da,last,ps = 0,0,[]
    # partocc = [[0 for _=1:max_sheet] for _=1:num_tri]
    partocc = [0 for _=1:max_sheet]
    for (i,tri)=enumerate(triangles)
        pts = barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[tri],:]')

        if num_cores == 1
            vals = eval_epm(pts,epm,sheets=max_sheet)
        else
            vals = reduce(hcat,pmap(x->eval_epm(x,epm,sheets=max_sheet),
                [pts[:,i] for i=1:size(pts,2)]))
        end

        for sheet = 1:max_sheet
            if all((vals[sheet,:] .< ebs.fermilevel) .| isapprox.(vals[sheet,:],epm.fermilevel))
                partocc[sheet] = 0
            elseif all((vals[sheet,:] .> ebs.fermilevel) .| isapprox.(vals[sheet,:],epm.fermilevel))
                partocc[sheet] = 2
            else
                partocc[sheet] = 1
            end
        end
         
        da = simplex_size(ebs.mesh.points[ebs.simplicesᵢ[tri],:]')/ndivs^2
        last = findlast(x->x==0,partocc)
        if last != nothing 
            sigma_be[i] = 2*length(epm.pointgroup)*sum(vals[1:last,:])*da
        end
        ps = findall(x->x==1,partocc)
        if length(ps) != 0
            part_be[i] = 2*length(epm.pointgroup)*sum(filter(x->(x<ebs.fermilevel || isapprox(x,ebs.fermilevel)), 
                    vals[ps,:]))*da
        end
    end
      
    approx_be = sum(sigma_be) + sum(part_be)
    approx_err = round(abs(approx_be - epm.bandenergy),sigdigits=3)
    println("The band energy error with triangular integration is $(approx_err)")
    sigma_be,part_be
end

@doc """
    containment_percentage(epm,ebs,divs,atol)

Calculate the containment percentage of a quadratic interval representation of the BandStructure.

# Arguments
- `epm::Union{EPM,EPM2D}`: an empirical pseudopotential model.
- `ebs::BandStructure`: a `BandStructure` data structure.
- `divs::Integer`: the number of divisions of a triangle when sampling quadratic surfaces
- `atol::Real=1e-6`: an absolute tolerance. Eigenvalues closer than `atol` to the
    interval are considered contained.

# Returns
- `percent::Real`: the percentage of eigenvalues contained by interval quadratics.
- `relerror::Vector{<:Real}`: the minimum distance of a eigenvalue to the interval
    quadratic divided by the size of the interval.

# Examples
```
using Pebsi.QuadraticIntegration, Pebsi.EPMs
epm = Al_epm
ebs = init_bandstructure(epm)
divs = 3
containment_percentage(epm,ebs,divs)
```
"""
function containment_percentage(epm::Union{EPM,EPM2D},
    ebs::BandStructure,divs::Integer,atol::Real=1e-6)
    dim = size(epm.recip_latvecs,1); deg = 2
    simplex_bpts = sample_simplex(dim,deg)
    bpts = sample_simplex(dim,divs)
    npts = size(bpts,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    relerror = zeros(2*length(ebs.simplicesᵢ)*size(bpts,2)*epm.sheets)
    rdiff₁ = 0; rdiff₂ = 0; interval = [0.,0.]; ind = 0
    vals₁ = zeros(npts); vals₂ = zeros(npts)
    for tri=1:length(ebs.simplicesᵢ)
        pts = barytocart(bpts,simplices[tri])
        evals = eval_epm(pts,epm,rtol=ebs.rtol,atol=ebs.atol)
        for sheet=1:epm.sheets
            interval = ebs.mesh_intcoeffs[tri][sheet]
            vals₁ = eval_poly(bpts,interval[1,:],dim,deg)
            vals₂ = eval_poly(bpts,interval[2,:],dim,deg)
            for i=1:npts
                ind = (2*i-1) + 2*(sheet-1)*npts + 2*(tri-1)*(epm.sheets*npts)
                rdiff₁ = abs(evals[sheet,i] - vals₁[i])/(vals₂[i] - vals₁[i])
                if rdiff₁ < 0 && !isapprox(rdiff₁,0,atol=atol)
                    relerror[ind] = -rdiff₁
                end
                rdiff₂ = (vals₂[i] - evals[sheet,i])/(vals₂[i] - vals₁[i])
                if rdiff₂ < 0 && !isapprox(rdiff₂,0,atol=atol)
                    relerror[ind+1] = -rdiff₂
                end
            end
        end
    end
    filter!(x -> x!=0, relerror)
    percent = (1-length(relerror)/(length(ebs.simplicesᵢ)*size(bpts,2)*epm.sheets))*100
    percent,relerror
end

do_nothing(x;dims=0) = x

@doc """
    quadlin_errest(epm,ebs)

Estimate the band energy error by taking the difference between quadratic and linear polynomials.
"""
function quadlin_esterr(epm,ebs;num_slices::Integer=def_num_slices)
    dim = size(epm.recip_latvecs,1)
    npg = length(epm.pointgroup)
    sbpt = sample_simplex(dim,2)
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    vals = [ebs.eigenvals[:,ebs.sym_unique[s]] for s=ebs.simplicesᵢ]
    nterms = if dim == 2 ntripts(2) else ntetpts(2) end
    bezpts = [[zeros(dim+1,nterms) for i=1:epm.sheets] for j=1:length(simplices)];
    lin_sigma_bandenergy = zeros(length(ebs.simplicesᵢ))
    lin_partial_bandenergy = [zeros(epm.sheets) for i=1:length(ebs.simplicesᵢ)];
    for j=1:length(ebs.simplicesᵢ)
        # Locate completely occupied sheets
        occsheets = findall(x->x==0, ebs.partially_occupied[j])
        # Values at the corners of the simplex
        v = vec(sum(vals[j][occsheets,:],dims=1))
        lin_sigma_bandenergy[j] = simplex_size(simplices[j])*mean(v)*2*npg
        for (i,p)=enumerate(ebs.partially_occupied[j])
            if p == 1            
                v = vals[j][i,:] .- ebs.fermilevel
                if dim == 2
                    bezpts = [barytocart(sbpt,simplices[j]); [v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,(v[2]+v[3])/2,v[3]]']
                else
                    bezpts = [barytocart(sbpt,simplices[j]); [v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,
                            (v[2]+v[3])/2,v[3],(v[1]+v[4])/2,(v[2]+v[4])/2,(v[3]+v[4])/2,v[4]]']
                end
                if dim == 2
                    fa = quad_area_volume(bezpts,"area")
                    lin_partial_bandenergy[j][i] = (ebs.fermilevel*fa + quad_area_volume(bezpts,"volume"))*2*npg
                else
                    fa = simpson3D(bezpts[end,:], simplices[j],"area")
                    lin_partial_bandenergy[j][i] = (ebs.fermilevel*fa + simpson3D(bezpts[end,:], simplices[j],"volume",
                        num_slices=def_num_slices))*2*npg
                end
            end 
        end
    end
    sum(abs.(ebs.sigma_bandenergy - lin_sigma_bandenergy)) + 
        sum(abs.(reduce(hcat,ebs.partial_bandenergy - lin_partial_bandenergy)))
end

@doc """
    cubequad_esterr(epm,ebs)

Estimate the band energy error by taking the difference between cubic and quadratic polynomials.
"""
function cubequad_esterr(epm,ebs)
    ebs3 = ebs; ebs3.polydegree = 3
    ns = length(ebs3.simplicesᵢ)
    c = [get_intercoeffs(i,mesh=ebs3.mesh,ext_mesh=ebs3.ext_mesh,sym_unique=ebs3.sym_unique,
        eigenvals=ebs3.eigenvals,simplicesᵢ=ebs3.simplicesᵢ,degree=ebs3.polydegree,constrained=false)
            for i=1:ns]
    coeffs = [i[2] for i=c]; intervals = [i[1] for i=c]
    ebs3.mesh_bezcoeffs = coeffs; ebs3.mesh_intcoeffs = intervals
    fl = ebs3.fermilevel
    be = calc_fabe(ebs3, quantity="volume", ctype="mean", fl=fl, num_slices=def_num_slices,sum_fabe=true)
    npg = length(epm.pointgroup); fl = ebs3.fermilevel; fa = epm.fermiarea
    abs(ebs.bandenergy - 2*npg*(be + fl*fa/npg))
end

@doc """
    kpoint_weights(epm,ebs;def_num_slices)

Calculate the k-points weights for a give approximation of the band structure.

# Arguments
- `epm::Union{EPM,EPM2D,EPM}`: the empirical pseudopotential.
- `ebs::BandStructure`: the band structure
- `num_slices::Integer=def_num_slices)`: the number of slices for integration.

# Returns
- `eigenvals::Matrix{<:Real}`: the eigenvalues of the band structure at each 
    sample k-point.
- `ns_weights::Matrix{<:Real}`: the weights for the number of states
- `be_weights::Matrix{<:Real}`: the weights for the band energy

# Examples
using Pebsi.EPMs, Pebsi.QuadraticIntegration
ebs = init_bandstructure(m11)
kpoint_weights(epm,ebs)
"""
function kpoint_weights(epm::Union{EPM,EPM2D,EPM},
        ebs::BandStructure; num_slices::Integer=def_num_slices)
    dim = size(epm.recip_latvecs,1)
    npg = length(epm.pointgroup)

    # The "true" Fermi level for the given approximation of the band structure
    fl = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="mean", num_slices=num_slices)

    # Calculate the Fermi area and band energy for each tile and sheet.
    mesh_fa = npg.*calc_fabe(ebs, quantity="area", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=false)
    mesh_be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=false)
    mesh_be = 2 .* (npg.*mesh_be .+ fl*mesh_fa);
    start = 2^dim +1
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ];
    spts = [[] for i=1:length(simplices)]
    for i=1:size(ebs.points,2)
        for j=1:length(simplices)
            if insimplex(carttobary(ebs.points[:,i],simplices[j]))
                spts[j] = [spts[j]; i]
            end
        end
    end
    
    nsheets,npts = size(ebs.eigenvals)
    ns_weights = zeros(nsheets,npts)
    be_weights = zeros(nsheets,npts);
    for i=1:length(spts)
        for j=1:nsheets
            ns_weights[j,spts[i]] .+= mesh_fa[i][j]/length(spts[i])
            be_weights[j,spts[i]] .+= mesh_be[i][j]/length(spts[i])
        end
    end


    # Integration is a weighted sum of eigenvalues. Remove the eigenvalues for 
    # the bounding box and divide the weights by the eigenvalues
    be_weights = (be_weights ./ ebs.eigenvals)[:,start:end]
    ns_weights = (ns_weights ./ ebs.eigenvals)[:,start:end]
    eigenvals = ebs.eigenvals[:,start:end]

    eigenvals,ns_weights, be_weights 
end
