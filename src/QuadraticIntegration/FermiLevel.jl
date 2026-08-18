@doc """
    calc_fl(epm,ebs;num_slices,window,ctype,fermi_area,test)

Calculate the Fermi level for a representation of the band structure.

# Arguments
- `epm::Union{EPM,EPM2D}`: an empirical pseudopotential 
- `ebs::BandStructure`: a `BandStructure` data structure.
- `num_slices::Int=10`: the number of slices for integration in 3D.
- `window::Union{Nothing,Vector{<:Real}}=ebs.fermilevel_interval`: an energy window
    that bounds the Fermi level.
- `ctype="mean"`: determines the coefficients that are used to compute the Fermi level.
   Options include: "mean"- use the coefficients obtained from the least-squares fit,
   "min"- use the lower coefficients of the interval coefficients and "max"- use 
   the upper coefficients of the interval coefficients.
- `fermi_area::Real=epm.fermiarea/length(epm.pointgroup)`: the sum of the areas of
    the shadows of the sheets.
- `test::Bool`: used to test the efficiency of the root-finding algorithm.

# Returns
- `E::Real`: the Fermi level for the quadratic approximation of the band structure.

# Examples
```jldoctest
using Pebsi.EPMs: m21
using Pebsi.QuadraticIntegration: init_bandstructure, calc_fl
epm = m21
ebs = init_bandstructure(epm);
abs(calc_fl(epm,ebs) - epm.fermilevel) < 1e-3
# output
true
```
"""
function calc_fl(epm::Union{EPM,EPM2D},ebs::BandStructure; 
    num_slices::Int=def_num_slices, window::Vector{<:Real}=ebs.fermilevel_interval, 
        ctype::String="mean", fermi_area::Real=epm.fermiarea/length(epm.pointgroup),
        test::Bool=false)

    if !(ctype in ["min","max","mean"])
        error("Invalid ctype.")
    end
    dim = size(epm.recip_latvecs,1) 
    maxsheet = round(Int,epm.electrons/2) + 2
    fermi_area = epm.fermiarea/length(epm.pointgroup)
    iters = 0    
    estart = if dim == 2 5 else 9 end # Don't consider points at the corners of the box
    if window == [0,0]
        E₁ = minimum(ebs.eigenvals[1,estart:end])
        E₂ = maximum(ebs.eigenvals[maxsheet,estart:end])
    else
        E₁,E₂ = window
    end

    # Make sure the window contains the approx. Fermi level.
    dE = 2*abs(E₂ - E₁)
    f₁ = calc_fabe(ebs, quantity="area", ctype=ctype,fl=E₁, num_slices=num_slices) - fermi_area
    iters₁ = 1
    while f₁ > 0
        if iters₁ > def_fl_max_iters
            error("Failed to calculate an upper limit for the Fermi level calculation after $(def_fl_max_iters) iterations.")
        end
        iters₁ += 1; E₁ -= dE; dE *= 2
        if iters₁ > def_fl_max_iters || dE == 0
            E₁ = minimum(ebs.eigenvals[1,:5:end])
        end
        f₁ = calc_fabe(ebs, quantity="area", ctype=ctype, fl=E₁, num_slices=num_slices) - fermi_area
    end
    dE = 2*abs(E₂ - E₁)
    f₂ = calc_fabe(ebs, quantity="area", ctype=ctype, fl=E₂, num_slices=num_slices) - fermi_area
    iters₂ = 1
    while f₂ < 0
        if iters₂ > def_fl_max_iters
            error("Failed to calculate an upper limit for the Fermi level calculation after $(def_fl_max_iters) iterations.")
        end
        iters₂ += 1; E₂ += dE; dE *= 2
        if iters₂ > def_fl_max_iters || dE == 0
            E₂ = maximum(ebs.eigenvals[maxsheet,5:end])
        end
        f₂ = calc_fabe(ebs, quantity="area", ctype=ctype, fl=E₂,num_slices=num_slices) - fermi_area
    end
    E = (E₁ + E₂)/2
    f₃,E₃,iters,f,t = 0,0,0,1e9,0
    ϵ = def_chandrupatla_tol
    fhist = []
    count = 1
    while abs(f) > ebs.fermiarea_eps
        count += 1
        iters += 1
        if (iters > def_fl_max_iters) || ((length(fhist) - length(unique(fhist))) > 5)
            @warn "Failed to converge the Fermi area to within the provided tolerance of $(ebs.fermiarea_eps) after $(count) iterations. Fermi area converged to within $(f)."
                break
        end
        f = calc_fabe(ebs, quantity="area", ctype=ctype, fl=E, num_slices=num_slices) - fermi_area
        append!(fhist,f)

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
        if ebs.fermilevel_method == fl_bisection
            t = 0.5
        # Chandrupatla method
        elseif ebs.fermilevel_method == fl_chandrupatla            
            ϕ₁ = (f₁ - f₂)/(f₃ - f₂)
            ξ₁ = (E₁ - E₂)/(E₃ - E₂)
            if 1 - √(1 - ξ₁) < ϕ₁ && ϕ₁ < √ξ₁
                α = (E₃ - E₁)/(E₂ - E₁)
                t = (f₁/(f₁ - f₂))*(f₃/(f₃ - f₂)) - α*(f₁/(f₃ - f₁))*(f₂/(f₂ - f₃))
            else
                t = 0.5
            end
            if t < ϵ
                t = ϵ
            elseif t > 1-ϵ
                t = 1-ϵ
            end
        else
            ArgumentError("The method for calculating the Fermi is either 1 or 2.")
        end
        E = E₁ + t*(E₂ - E₁)
    end
    if test 
        (iters₁ + iters₂ + iters, E)
    else
        E
    end
end

@doc """
    calc_flbe!(epm,ebs;num_slices)

Calculate the Fermi level and band energy for a given rep. of the band struct.

# Arguments
- `epm::Union{EPM2D,EPM}`: an empirical pseudopotential.
- `ebs::BandStructure`: the band structure container.
- `num_slices::Integer=def_num_slices`: the number of slices when integrating in 3D.
- `flerrors::Bool=true`: if true, band energy errors include effects from Fermi 
    level error.

# Returns
- `ebs::BandStructure`: updated values within container for the band energy error,
    Fermi area error, Fermi level interval, Fermi area interval, band energy
    interval, and the partially occupied sheets.

# Examples
```jldoctest
using Pebsi.EPMs: m21
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!
epm = m21
ebs = init_bandstructure(epm);
calc_flbe!(epm,ebs)
abs(ebs.bandenergy - epm.bandenergy) < 1e-2
# output
true
```
"""
function calc_flbe!(epm::Union{EPM2D,EPM},ebs::BandStructure;
    num_slices::Integer=def_num_slices, flerrors::Bool=true)::BandStructure
     
    # The number of point operators
    npg = length(epm.pointgroup)
    if ebs.exactfit == true
        fl = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="mean", num_slices=num_slices)
        be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        # Calculate occupancy of sheets
        simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
        # The areas of the triangles in the mesh
        simplex_sizes = [simplex_size(s) for s=simplices];     
        nsheet,npts = size(ebs.eigenvals);
        mesh_fa = calc_fabe(ebs, quantity="area", ctype="mean", fl=ebs.fermilevel, num_slices=def_num_slices, sum_fabe=false)
        partial_occ = [[(
            if isapprox(mesh_fa[tri][sheet],0,atol=ebs.atol)
                2
            elseif isapprox(mesh_fa[tri][sheet],simplex_sizes[tri],atol=ebs.atol)
                0
            else
                1
            end
            ) for sheet=1:epm.sheets] for tri = 1:length(simplices)]
            
        ebs.fermilevel = fl; ebs.bandenergy = 2*npg*(be + fl*epm.fermiarea/npg)
        ebs.partially_occupied = partial_occ
        return ebs
    end
    
    dim = size(epm.recip_latvecs,1)
    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(dim,2)
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The areas of the triangles in the mesh
    simplex_sizes = [simplex_size(s) for s=simplices]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    # The number of simplices
    ns = length(ebs.simplicesᵢ)
    # The "true" Fermi level for the given approximation of the band structure
    fl = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="mean", num_slices=num_slices)
    # The larger Fermi level computed with the upper limit of approximation intervals
    fl₁ = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="max", num_slices=num_slices)
    # The smaller Fermi level computed with the lower limit of approximation intervals
    fl₀ = calc_fl(epm, ebs, fermi_area=epm.fermiarea/npg, ctype="min", num_slices=num_slices)
    # The "true" Fermi area for each quadratic triangle (triangle and sheet) for the
    # given approximation of the BandStructure
    mesh_fa = calc_fabe(ebs, quantity="area", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=false)
     
    # The smaller Fermi area for each quadratic triangle using the upper limit of the approximation
    # intervals with the lower limit of the Fermi level interval
    if !flerrors
        mesh_fa₁ = calc_fabe(ebs, quantity="area", ctype="min", fl=fl, num_slices=num_slices,
            sum_fabe=false)
        mesh_fa₀ = calc_fabe(ebs, quantity="area", ctype="max", fl=fl, num_slices=num_slices,
            sum_fabe=false)
    # The larger Fermi area for each quadratic triangle using the lower limit of the approximation
    # intervals with the upper limit of the Fermi level interval
    else
        mesh_fa₁ = calc_fabe(ebs, quantity="area", ctype="min", fl=fl₁, num_slices=num_slices,
            sum_fabe=false)
        mesh_fa₀ = calc_fabe(ebs, quantity="area", ctype="max", fl=fl₀, num_slices=num_slices,
            sum_fabe=false)
    end

    # The smaller and larger Fermi areas or the limits of the Fermi area interval
    fa₀,fa₁ = sum(sum(mesh_fa₀)),sum(sum(mesh_fa₁))
    # The "true" band energy for the given approximation of the band structure
    be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=true)

    # The Fermi area errors for each quadratic triangle (triangle and sheet) for the
    # given band structure approximation
    mesh_fa_errs = mesh_fa₁ .- mesh_fa₀
     
    # Determine which triangles and sheets are partially occupied by comparing
    # the Fermi area (shadows of the sheets) for each quadratic triangle to zero
    # and the area of the triangle.
    partial_occ = [[(
        if (isapprox(mesh_fa₁[tri][sheet],0,atol=ebs.atol) &&
            isapprox(mesh_fa₀[tri][sheet],0,atol=ebs.atol))
            2
        elseif (isapprox(mesh_fa₁[tri][sheet],simplex_sizes[tri],atol=ebs.atol) &&
            isapprox(mesh_fa₀[tri][sheet],simplex_sizes[tri],atol=ebs.atol))
            0
        else
            1
        end
    ) for sheet=1:epm.sheets] for tri = 1:ns]
     
    # Determine which triangle and sheets are occupied, partially occupied, or 
    # unoccupied for the "true" approximation of the BandStructure (least-squares
    # fitting of eigenvalues).
    true_partial_occ = [[(
        if isapprox(mesh_fa[tri][sheet],0,atol=ebs.atol)
            2
        elseif isapprox(mesh_fa[tri][sheet],simplex_sizes[tri],atol=ebs.atol)
            0
        else
            1
        end
    ) for sheet=1:epm.sheets] for tri = 1:ns]
    
    # The largest index of sheets that are completely occupied (integer) for each triangle
    sigmas = [findlast(x->x==0,partial_occ[i]) for i=1:length(partial_occ)] 
    # The indices of sheets that are partially occupied (vector) for each triangle
    partials = [findall(x->x==1,partial_occ[i]) for i=1:length(partial_occ)] 
    true_sigmas = [findlast(x->x==0,true_partial_occ[i]) for i=1:length(true_partial_occ)]
    true_partials = [findall(x->x==1,true_partial_occ[i]) for i=1:length(true_partial_occ)] 
     
    # For each triangle, sum the eigenvalues of all sheets that are completely occupied
    # and calculate Bezier coeffients and intervals coefficients for the "sigma" sheet.
    nterms = sum([i for i=1:dim+1])
    sigma_intcoeffs = [
        (if sigmas[i] == nothing
            [[zeros(2,nterms)],[zeros(1,nterms)]]
        else
            get_intercoeffs(i,mesh=ebs.mesh,ext_mesh=ebs.ext_mesh,sym_unique=ebs.sym_unique,
            eigenvals=ebs.eigenvals,simplicesᵢ=ebs.simplicesᵢ,degree=ebs.polydegree,
            fatten=ebs.fatten,num_near_neigh=ebs.num_near_neigh,sigma=sigmas[i],
            epm=epm,neighbor_method=ebs.neighbor_method)
        end) for i=1:ns]

    # Calculate the "sigma" coefficients for the "true" occupations of the sheets. The true sigma 
    # coefficients and intervals and the regular coefficients and intervals are only different if the
    # occupations are different
    true_sigma_intcoeffs = [
        (if true_sigmas[i] == nothing
            [[zeros(2,nterms)],[zeros(1,nterms)]]
        else
            get_intercoeffs(i,mesh=ebs.mesh,ext_mesh=ebs.ext_mesh,sym_unique=ebs.sym_unique,
            eigenvals=ebs.eigenvals,simplicesᵢ=ebs.simplicesᵢ,degree=ebs.polydegree,
            fatten=ebs.fatten,num_near_neigh=ebs.num_near_neigh,sigma=true_sigmas[i],epm=epm,
            neighbor_method = ebs.neighbor_method)
        end) for i=1:ns]
    
    # Assign the sigma intervals and coefficients their own variables for both true and regular.
    sigma_intervals = [sigma_intcoeffs[i][1][1] for i=1:length(sigma_intcoeffs)]
    sigma_coeffs = [sigma_intcoeffs[i][2][1] for i=1:length(sigma_intcoeffs)]
     
    true_sigma_intervals = [true_sigma_intcoeffs[i][1][1] for i=1:length(true_sigma_intcoeffs)]
    true_sigma_coeffs = [true_sigma_intcoeffs[i][2][1] for i=1:length(true_sigma_intcoeffs)]
     
    # Calculate the contribution to the band energy of the sigma sheets in each triangle using the 
    # "true" coefficients and the "true" occupations.
    sigma_be = [simplex_size(simplices[i])*mean(true_sigma_coeffs[i]) for i=1:length(true_sigma_intcoeffs)]
     
    # Calculate a lower limit for the sigma contribution to the band energy using the regular 
    # occupations and the lower limit of the sigma interval coefficients.
    sigma_be₀ = [#sigma_intervals[i] == 0 ? 0 : 
        simplex_size(simplices[i])*mean(sigma_intervals[i][1,:]) for i=1:length(sigma_intcoeffs)]
    # Calculate the upper limit of the sigma contribution to the band energy using the regular
    # occupations and the upper limit of the sigma interval coefficients.
    sigma_be₁ = [#sigma_intervals[i] == 0 ? 0 : 
        simplex_size(simplices[i])*mean(sigma_intervals[i][2,:]) for i=1:length(sigma_intcoeffs)]
     
    # The contribution to the band energy error from the sigma sheets in each triangle
    sigma_be_errs = (sigma_be₁ - sigma_be₀)./2 # the average error
     
    # The "true" contribution to the band energy from the partially occupied sheets using the
    # least-squares coefficients and the true occupations. The leading term takes into account
    # the integral transform.
    partial_be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
        sum_fabe=false, sheets=true_partials)
    for (i,po)=enumerate(true_partials)
        partial_be[i][po] += fl*mesh_fa[i][po]
    end
 
    # The lower limit of the contribution to the band energy from the partially occupied sheets
    # obtained using the regular occupations, the lower limit of the interval coefficients,
    # and the upper limit of the Fermi level interval
    if !flerrors
        partial_be₀ = calc_fabe(ebs, quantity="volume", ctype="min", fl=fl,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₁ = calc_fabe(ebs, quantity="volume", ctype="max", fl=fl,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        for (i,po)=enumerate(partials)
            partial_be₀[i][po] += fl*mesh_fa[i][po]
            partial_be₁[i][po] += fl*mesh_fa[i][po] 
        end
    else
        partial_be₀ = fl₀*mesh_fa₀ + calc_fabe(ebs, quantity="volume", ctype="max", fl=fl₀,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        partial_be₁ = fl₁*mesh_fa₁ + calc_fabe(ebs, quantity="volume", ctype="min", fl=fl₁,
            num_slices=num_slices, sum_fabe=false, sheets=partials)
        for (i,po)=enumerate(partials)
            partial_be₀[i][po] += fl₀*mesh_fa₀[i][po]
            partial_be₁[i][po] += fl₁*mesh_fa₁[i][po] 
        end
    end

    # The contributions to the band energy error from the partially occupied quadratic triangles
    # part_be_errs = [sum(abs.(p)) for p=(partial_be₁ .- partial_be₀)./2] # average error
    part_be_errs = (partial_be₁ .- partial_be₀)./2

    # The limits of the Fermi area interval
    ebs.fermiarea_interval = npg.*[fa₀,fa₁]
    # The Fermi area errors in each triangle
    ebs.fermiarea_errors = npg*[sum(m) for m=mesh_fa_errs]
    ebs.fermilevel_interval = [fl₀,fl₁]
    ebs.fermilevel = fl
      
    # The limits of the band energy interval
    be₀ = 2*npg*(sum(sigma_be₀) + sum(sum(partial_be₀))) 
    be₁ = 2*npg*(sum(sigma_be₁) + sum(sum(partial_be₁)))
    ebs.bandenergy_interval = [be₀,be₁]
    ebs.bandenergy = 2*npg*(be + fl*epm.fermiarea/npg)

    ebs.partially_occupied = partial_occ
    ebs.sigma_bandenergy = 2*npg.*sigma_be
    ebs.partial_bandenergy = 2*npg.*partial_be

    ebs.bandenergy_sigma_errors = 2*npg.*sigma_be_errs
    ebs.bandenergy_partial_errors = 2*npg.*part_be_errs
    # ebs.bandenergy_errors = 2*npg.*(sigma_be_errs + part_be_errs)
    ebs.bandenergy_errors = 2*npg.*(sigma_be_errs + [sum(abs.(p)) for p=part_be_errs])
    ebs
end

@doc """
    get_tolerances(epm,ebs)

Calculate the Fermi level and Fermi area tolerances.

# Arguments
- `epm::Union{EPM2D,EPM}`: an empirical pseudopotential.
- `ebs::BandStructure`: a quadratic approximation of the band structure.

# Returns
- `db::Real`: the derivative of the band energy with respect to the Fermi level.
- `da::Real`: the derivative of the Fermi area with respect to the Fermi level.
- `dltol::Real`: the Fermi level tolerance.
- `datol::Real`: the Fermi area tolerance.

# Examples
```jldoctest
using Pebsi.EPMs: m41
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!, get_tolerances
epm = m41
ebs = init_bandstructure(epm);
calc_flbe!(epm,ebs)
tol = get_tolerances(epm,ebs)
length(tol)
# output
4
```
"""
function get_tolerances(epm,ebs; num_slices=def_num_slices)
    dim = size(epm.recip_latvecs,1)
    start = 2^dim + 1
    stepsize = (maximum(ebs.eigenvals[:,start:end]) - 
       minimum(ebs.eigenvals[:,start:end]))*def_deriv_step
    numsteps = 4 # number of band energies/Fermi areas needed to compute derivatives
    es = collect(-numsteps/2*stepsize:stepsize:numsteps/2*stepsize) .+ ebs.fermilevel
    # Sample points within the triangle for a quadratic in barycentric coordinates.
    simplex_bpts = sample_simplex(dim,2) 
    # The triangles in the triangular mesh
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    # The six sample points in each triangle for a quadratic polynomial
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]

    npg = length(epm.pointgroup); bens = []; fas = []
    for fl=es
        be = calc_fabe(ebs, quantity="volume", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        fa = calc_fabe(ebs, quantity="area", ctype="mean", fl=fl, num_slices=num_slices,
            sum_fabe=true)
        # be = 2*npg*(be + fl*epm.fermiarea/npg)
        # Removed translation of band energy because numbers close to zero throw off the derivative
        be = 2*npg*be
        push!(bens,be); push!(fas,fa)
    end
    dbs = [(-bens[i+2] + 8*bens[i+1] - 8*bens[i-1] + bens[i-2])/(es[i+2]-es[i-2]) for i=3:length(bens)-2]
    das = [(-fas[i+2] + 8*fas[i+1] - 8*fas[i-1] + fas[i-2])/(es[i+2]-es[i-2]) for i=3:length(fas)-2];
    db = maximum(abs.(das)); da = maximum(abs.(dbs))
    dltol = ebs.target_accuracy/db; datol = da*dltol
    db,da,dltol,datol
end

@doc """
    calc_fabe(ebs,quantity,ctype,fl;num_slices,sum_fabe,sheets)

Calculate the Fermi area or band energy for a candidate Fermi level per patch or the sum.

# Arguments
- `ebs::BandStructure`: a `BandStructure` composite type.
- `quantity::String`: the quantity calculated ("area" or "volume")
- `ctype::String`: determines which coefficients to use. Options include "min",
    "max", and "mean". The least-squares coefficients are used with option "mean".
- `fl::Real=ebs.fermilevel`: the candidate Fermi level.
- `num_slices::Integer=def_num_slices`: the number of slices for integration in 3D.
- `sum_fabe::Bool=true`: if true, sum the Fermi area or band energy of all patches.
- `sheets::Vector:` the sheets per triangle to consided as a vector of vectors.

# Returns
- `fabe::Real`: the Fermi level or band energy

# Examples
```
using Pebsi.EPMs, Pebsi.QuadraticIntegration
epm = m11
ebs = init_bandstructure(epm)
quantity = "area"
fl = epm.fermilevel
ctype = "mean"
calc_fabe(ebs,quantity,ctype,fl)
```
"""
function calc_fabe(ebs::BandStructure; quantity::String, ctype::String, fl::Real=ebs.fermilevel,
    num_slices::Integer=def_num_slices, sum_fabe::Bool=true, sheets::Union{Nothing,Vector}=nothing)
     
    ns = length(ebs.simplicesᵢ)
    nsheets = size(ebs.eigenvals,1)
    if sheets == nothing
        sheets = [collect(1:nsheets) for i=1:ns]
    end
    dim = length(ebs.simplicesᵢ[1]) - 1
    simplices = [Matrix(ebs.mesh.points[s,:]') for s=ebs.simplicesᵢ]
    if ctype == "min" || ctype == "max"
        coeffs = ebs.mesh_intcoeffs
        cfun = if ctype == "min" minimum else maximum end
    else
        coeffs = ebs.mesh_bezcoeffs
        cfun = do_nothing
    end
     
    fabe = [[0. for sheet=1:nsheets] for tri=1:ns]
    quadfun = (
        if dim == 2
            if ebs.polydegree > 2 area_volume2D else quad_area_volume end
        else
            if ebs.polydegree > 3 volume_hypvol3D else simpson3D end
        end)
     
    d = if ebs.polydegree < 2 2 else ebs.polydegree end       
    simplex_bpts = sample_simplex(dim,d)
    simplex_pts = [barytocart(simplex_bpts,s) for s=simplices]
    for tri=1:ns
        for sheet=sheets[tri]
            # println("bezpts: ", [simplex_pts[tri]; [cfun(coeffs[tri][sheet],dims=1)...]' .- fl])
            # println("quantity: ", quantity)
            # println("num_slices: ", num_slices)
            fabe[tri][sheet] = quadfun([simplex_pts[tri]; [cfun(coeffs[tri][sheet],dims=1)...]' .- fl],
                quantity, num_slices=num_slices)
        end
    end
    if sum_fabe sum(sum(fabe)) else fabe end
end
