@doc """
    BandStructure

A container for all variables related to the band structure.

# Arguments
- `init_msize::Int`: the initial size of the mesh over the IBZ. The number
    of points is approximately proportional to init_msize^2/2.
- `num_near_neigh::Int`: the number of nearest neighbors to consider when 
    calculating interval coefficients. For example, a value of 1 will include 
    neighbors that are a distance of 1 away or are connected by an edge with the
    corners of the simplex.
- `num_neighbors::Int`: the minimum number of neighbors to include in the 
    calculation of interval coefficients. 
- `fermiarea_eps::Float64`: a tolerance used during the bisection algorithm that 
    determines how close the midpoints of the Fermi area interval is to the true
    Fermi area.
- `target_accuracy::Float64`: the accuracy desired for the band energy at the end 
    of the computation.
- `fermilevel_method::FermiLevelMethod`: the root-finding method for computing the
    Fermi level: `fl_bisection` or `fl_chandrupatla`.
- `refine_method::RefineMethod`: the method of refinement: `refine_most_error`,
    `refine_above_allowed`, `refine_fraction_above_allowed`,
    `refine_fermiarea_above_allowed`, `refine_fermiarea_largest`,
    `refine_partially_occupied` or `refine_largest_fraction`.
- `sample_method::SampleMethod`: the method of sampling a tile with too much
    error: `sample_center`, `sample_edge_midpoints` or `sample_adaptive`.
- `neighbor_method::NeighborMethod`: the method for selecting neighboring points
    in the calculation of interval coefficients: `neighbors_closest`,
    `neighbors_surrounding` or `neighbors_inside`.
- `rtol::Float64`: a relative tolerance for floating point comparisons.
- `atol::Float64`: an absolute tolerance for floating point comparisons.
- `mesh::PyObject`: a Delaunay triangulation of points over the IBZ.
- `simplicesᵢ::Vector{Vector{Int}}`: the indices of points at the corners of the 
    tile for all tiles in the triangulation.
- `ext_mesh::PyObject`: a Delaunay triangulation of points within and around the 
    IBZ. The number of points outside is determined by `num_near_neigh`.
- `sym_unique::AbstractVector{<:Integer}`: the indices of symmetrically unique points
    in the mesh.
- `eigenvals::AbstractMatrix{<:Real}`: the eigenvalues at each of the points unique
    by symmetry.
- `fatten::Float64` a variable that scales the size of the interval coefficients.
- `mesh_intcoeffs::Vector{Vector{Matrix{Float64}}}`:the interval Bezier 
    coefficients for all tiles and sheets.
- `mesh_bezcoeffs::Vector{Vector{Vector{Float64}}}`: the least-squares Bezier
    coefficients for all tiles and sheets.
- `fermiarea_interval::AbstractVector{<:Real}`: the Fermi area interval. 
- `fermilevel_interval::AbstractVector{<:Real}`: the Fermi level interval. 
- `bandenergy_interval::AbstractVector{<:Real}`: the band energy interval.
- `fermilevel::Real`: the approximate Fermi level.
- `bandenergy::Real`: the approximate band energy.
- `sigma_bandenergy::Vector{<:Real}`: the sigma band energy (the energy of the 
    approximate sheets that are completely below the approximate Fermi level) 
    for each simplex in the mesh.
- `partial_bandenergy::Vector`: the partial band energy (the energy of 
    the approximate sheets that are below and above the approximate Fermi level)
    for each simplex in the mesh.
- `partially_occupied::Vector{Vector{Int64}}`: the sheets that are partially 
    occupied in each tile.
- `bandenergy_errors::Vector{<:Real}`: estimates of the band energy errors in 
    each tile in the mesh.
- `bandenergy_sigma_errors::Vector{<:Real}`: band energy errors from sigma sheets.
- `bandenergy_partial_errors::Vector`: band energy errors from partial
    sheets.
- `fermiarea_errors::AbstractVector{<:Real}`: the Fermi area errors for each tile 
    in the triangulation.
- `weighted::bool`: calculate the interval coefficients using weighted least squares
    if true.
- `constrained::bool`: calculate the interval coefficients with constrained least
    squares if true.
- `stop_criterion::StopCriterion`: determines the criterion used to stop adaptive
    refinement.
    `stop_total_error`: The sum of the estimated band energy errors.
    `stop_energy_change`: The difference in band energy between two AMR iterations
      is less than the band energy accuracy goal.
    `stop_interval`: db/da*Δl is less than the band energy accuracy where db is the derivative of
      the band energy with respect to the Fermi level, da is the derivative of the 
      Fermi area with respect to the Fermi level, and Δl is the uncertainty of the
      Fermi level.
    `stop_kpoint_target`: The number of k-points is close to, or greater than, the
      desired number of k-points.
- `target_kpoints::Int`: the desired number of k-points for the calculation.
    This may be ignored depending on `stop_criterion`.
- `exactfit::Bool`: the polynomial fit goes through the eigenvalues if true.
- `polydegree::Int`: the degree of the polynomial
"""
mutable struct BandStructure
    init_msize::Int
    num_near_neigh::Int
    num_neighbors::Int
    fermiarea_eps::Float64
    target_accuracy::Float64
    fermilevel_method::FermiLevelMethod
    refine_method::RefineMethod
    sample_method::SampleMethod
    neighbor_method::NeighborMethod
    rtol::Float64
    atol::Float64
    mesh::PyObject
    points::Matrix{Float64}
    simplicesᵢ::Vector{Vector{Int}}
    ext_mesh::PyObject
    sym_unique::Vector{Int}
    eigenvals::Matrix{Float64}
    fatten::Float64
    mesh_intcoeffs::Vector{Vector{Matrix{Float64}}}
    mesh_bezcoeffs::Vector{Vector{Vector{Float64}}}
    fermiarea_interval::Vector{Float64}
    fermilevel_interval::Vector{Float64}
    bandenergy_interval::Vector{Float64}
    fermilevel::Float64
    bandenergy::Float64
    sigma_bandenergy::Vector{Float64}
    partial_bandenergy::Vector{Vector{Float64}}
    partially_occupied::Vector{Vector{Int64}}
    bandenergy_errors::Vector{Float64}
    bandenergy_sigma_errors::Vector{Float64}
    bandenergy_partial_errors::Vector{Vector{Float64}}
    fermiarea_errors::Vector{Float64}
    weighted::Bool
    constrained::Bool
    stop_criterion::StopCriterion
    target_kpoints::Int
    exactfit::Bool
    polydegree::Int
end

@doc """
    init_bandstructure(epm,init_msize,init_num_kpoints,num_near_neigh,num_neighbors,fermiarea_eps,
        target_accuracy,fermilevel_method,refine_method,sample_method,neighbor_method,
        fatten,weighted,constrained,stop_criterion,target_kpoints,exactfit,polydegree,
        rtol,atol)

Initialize a band structure container.

# Arguments
- `epm::Union{EPM,EPM2D}`: an empirical pseudopotential.

See the documentation for `BandStructure` for a description of the remaining arguments.

# Returns
- `::BandStructure`: a container for information on the band structure approximation.

# Examples
```jldoctest
import Pebsi.EPMs: m11
import Pebsi.QuadraticIntegration: init_bandstructure,BandStructure
ebs = init_bandstructure(m11)
typeof(ebs)
# output
BandStructure
```
"""
function init_bandstructure(
    epm::Union{EPM,EPM2D,EPM};
    init_msize::Integer=def_init_msize,
    init_num_kpoints::Integer=def_num_kpoints,
    num_near_neigh::Integer=def_num_near_neigh,
    num_neighbors::Union{Nothing,Integer}=nothing,
    fermiarea_eps::Real=def_fermiarea_eps,
    target_accuracy::Real=def_target_accuracy,
    fermilevel_method::FermiLevelMethod=def_fermilevel_method,
    refine_method::RefineMethod=def_refine_method,
    sample_method::SampleMethod=def_sample_method,
    neighbor_method::NeighborMethod=def_neighbor_method,
    fatten::Real=def_fatten,
    weighted::Bool=def_weighted,
    constrained::Bool=def_constrained,
    stop_criterion::StopCriterion=def_stop_criterion,
    target_kpoints::Integer=def_target_kpoints,
    exactfit::Bool=false, 
    polydegree::Integer=2,
    rtol::Real=def_rtol,
    atol::Real=def_atol)

    dim = size(epm.recip_latvecs,1)
    if exactfit
        (points, mesh, simplicesᵢ, sym_unique, eigenvals, mesh_bezcoeffs, mesh_intcoeffs) =
            init_exactfit(epm,num_kpoints=init_num_kpoints,polydegree=polydegree,atol=atol,
            rtol=rtol)
        ext_mesh = mesh; num_neighbors = 0;
    else
        mesh = ibz_init_mesh(epm.ibz,init_msize;rtol=rtol,atol=atol)
        mesh,ext_mesh,sym_unique = get_extmesh(epm.ibz,mesh,epm.pointgroup,
            epm.recip_latvecs,num_near_neigh; rtol=rtol,atol=atol)
        simplicesᵢ = notbox_simplices(mesh)
        uniqueᵢ = sort(unique(sym_unique))[2:end]
        estart = if dim == 2 4 else 8 end
        eigenvals = zeros(Float64,epm.sheets,estart+length(uniqueᵢ))
        for i=uniqueᵢ
            eigenvals[:,i] = eval_epm(mesh.points[i,:], epm, rtol=rtol, atol=atol)
        end
        if num_neighbors == nothing
            num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
        end
        coeffs = [get_intercoeffs(index,mesh=mesh,ext_mesh=ext_mesh,sym_unique=sym_unique,
            eigenvals=eigenvals,simplicesᵢ=simplicesᵢ,degree=polydegree,fatten=fatten,
            num_near_neigh=num_near_neigh,epm=epm,neighbor_method=neighbor_method,
            num_neighbors=num_neighbors, weighted=weighted, constrained=constrained) 
            for index=1:length(simplicesᵢ)]
        mesh_intcoeffs = [coeffs[index][1] for index=1:length(simplicesᵢ)]
        mesh_bezcoeffs = [coeffs[index][2] for index=1:length(simplicesᵢ)]
        points = Matrix(mesh.points')
    end
    
    partially_occupied = [zeros(Int,epm.sheets) for _=1:length(simplicesᵢ)]
    bandenergy_errors = zeros(length(simplicesᵢ))
    bandenergy_sigma_errors = zeros(length(simplicesᵢ))
    bandenergy_partial_errors = [zeros(epm.sheets) for i=1:length(simplicesᵢ)]
    fermiarea_errors = zeros(length(simplicesᵢ))
    sigma_bandenergy = zeros(length(simplicesᵢ))
    partial_bandenergy = [zeros(epm.sheets) for i=1:length(simplicesᵢ)]
    if exactfit
        fermilevel_interval = [minimum(eigenvals), maximum(eigenvals)]
    else
        fermilevel_interval=[0,0]
    end
    fermiarea_interval=[0,0]; bandenergy_interval=[0,0]; fermilevel=0; bandenergy=0
        
    BandStructure(
        init_msize,
        num_near_neigh,
        num_neighbors,
        fermiarea_eps,
        target_accuracy,
        fermilevel_method,
        refine_method,
        sample_method,
        neighbor_method,
        rtol,
        atol,
        
        mesh,
        points,
        simplicesᵢ,
        ext_mesh,
        sym_unique,
        eigenvals,
        fatten,
        mesh_intcoeffs,
        mesh_bezcoeffs,
        fermiarea_interval,
        fermilevel_interval,
        bandenergy_interval,
        fermilevel,
        bandenergy,
        sigma_bandenergy,
        partial_bandenergy,
        partially_occupied,
        bandenergy_errors,
        bandenergy_sigma_errors,
        bandenergy_partial_errors,
        fermiarea_errors,

        weighted,
        constrained,
        stop_criterion,
        target_kpoints,
        exactfit,
        polydegree)
end   

@doc """
    select_neighbors(neighborsᵢ,simplex,ext_mesh,dim,num_neighbors,neighbor_method,epm)

Narrow the candidate neighbours of a simplex to the ones used when fitting.

# Arguments
- `neighborsᵢ::AbstractVector{<:Integer}`: the candidate neighbours, as indices
    into `ext_mesh`.
- `simplex::AbstractMatrix{<:Real}`: the corners of the simplex in the columns of
    a matrix.
- `ext_mesh::PyObject`: a triangulation of the region within and around the IBZ.
- `dim::Integer`: the number of dimensions.
- `num_neighbors::Integer`: how many neighbours to keep.
- `neighbor_method::NeighborMethod`: which selection strategy to use.
- `epm`: an empirical pseudopotential, required only by `neighbors_inside`.

# Returns
- `::AbstractVector`: the indices of the neighbours to use. Empty for
    `neighbors_inside`, where the points come from a grid inside the simplex
    instead and the caller samples them directly.
"""
function select_neighbors(neighborsᵢ::AbstractVector,simplex::AbstractMatrix{<:Real},
    ext_mesh::PyObject,dim::Integer,num_neighbors::Integer,
    neighbor_method::NeighborMethod,epm)

    # Select neighbors that are closest to the triangle.
    if neighbor_method == neighbors_closest
        # Snapped to a multiple of the tolerance, as in choose_neighbors.
        edge = minimum([norm(simplex[:,i] - simplex[:,j])
            for i=1:size(simplex,2) for j=1:size(simplex,2) if i < j])
        dtol = def_neighbor_dist_rtol*edge
        dist = round.([minimum([norm(ext_mesh.points[i,:] - simplex[:,j]) for j=1:dim+1])
            for i=neighborsᵢ] ./ dtol) .* dtol
        neighborsᵢ[sortperm(dist, alg=Base.Sort.DEFAULT_STABLE)][1:num_neighbors]
    # Select neighbors that surround the triangle and are close to the triangle.
    elseif neighbor_method == neighbors_surrounding
        neighbors = Matrix(ext_mesh.points[neighborsᵢ,:]')
        if dim == 2
            choose_neighbors(simplex,neighborsᵢ,neighbors; num_neighbors=num_neighbors)
        else
            choose_neighbors3D(simplex,neighborsᵢ,neighbors; num_neighbors=num_neighbors)
        end
    # Neighbors are taken from a uniform grid within the triangle, so none of the
    # candidates are kept; the caller samples the interior instead.
    elseif neighbor_method == neighbors_inside
        if epm == nothing
            error("Must provide an EPM when computing neighbors within the triangle.")
        end
        []
    else
        error("Unhandled NeighborMethod: $(neighbor_method)")
    end
end

@doc """
    get_intercoeffs(index, mesh, ext_mesh, sym_unique, eigenvals, simplicesᵢ, degree,
        fatten, num_near_neigh; sigma, epm, neighbor_method, num_neighbors, weighted,
        constrained, atol)

Calculate the interval Bezier points for all sheets.

# Arguments
- `index::Integer`: the index of the simplex in `simplicesᵢ`.
- `mesh::PyObject`: a triangulation of the irreducible Brillouin zone.
- `ext_mesh::PyObject`: a triangulation of the region within and around the IBZ.
- `sym_unique::AbstractVector{<:Real}`: the index of the eigenvalues for each point
    in the `mesh`.
- `eigenvals::AbstractMatrix{<:Real}`: a matrix of eigenvalues for the symmetrically
    distinc points as columns of a matrix.
- `simplicesᵢ::AbstractVector`: the simplices of `mesh` that do not include the box points.
- `degree::Integer`: the degree of the polynomial.
- `fatten::Real=def_fatten`: scale the interval coefficients by this amount.
- `num_near_neigh::Integer=def_num_near_neigh`: how many nearest neighbors to include.
- `sigma::Integer=0`: the number of sheets summed and then interpolated, if any.
- `epm::Union{Nothing,EPM2D,EPM}=nothing`: an empirical pseudopotential.
- `neighbor_method::NeighborMethod=def_neighbor_method`: the method for calculating neighbors
    to include in the calculation.
- `num_neighbors::Union{Nothing,Integer}=nothing`: the minimum number of neighbors
    included in the calculation of interval coefficients.
- `weighted::Bool=false`: if true, points are weighted by their minimum distance to
    a boundary of the simplex.
- `constrained::Bool=true`: if true, use constrained least squares.
- `atol::Real=def_atol`: an absolume tolerance.

# Returns
- `inter_bezpts::Vector{Matrix{Float64}}`: the interval Bezier points for each sheet.
- `bezcoeffs::Vector{Vector{Float64}}`: the Bezier coefficients from the least-squares
    fit for each sheet.

# Examples
```jldoctest
import Pebsi.EPMs: m2ibz,m2pointgroup,m2recip_latvecs,m2rules,m2cutoff,eval_epm
import Pebsi.Mesh: ibz_init_mesh, get_extmesh, notbox_simplices
import Pebsi.QuadraticIntegration: get_intercoeffs
n = 10
mesh = ibz_init_mesh(m2ibz,n)
simplicesᵢ = notbox_simplices(mesh)
num_near_neigh = 2
mesh,ext_mesh,sym_unique = get_extmesh(m2ibz,mesh,m2pointgroup,m2recip_latvecs,num_near_neigh)
sheets = 7
energy_conv = 1
eigenvals = zeros(sheets,size(mesh.points,1))
for i = sort(unique(sym_unique))[2:end]
    eigenvals[:,i] = eval_epm(mesh.points[i,:],m2recip_latvecs,m2rules,m2cutoff,sheets,energy_conv)
end
index = 1
degree = 2
intercoeffs,bezcoeffs = get_intercoeffs(index,mesh=mesh,ext_mesh=ext_mesh,sym_unique=sym_unique,
    eigenvals=eigenvals,simplicesᵢ=simplicesᵢ,degree=degree)
length(bezcoeffs)
# output
7
```
"""
function get_intercoeffs(index::Integer; mesh::PyObject, ext_mesh::PyObject,
    sym_unique::AbstractVector{<:Real}, eigenvals::AbstractMatrix{<:Real}, 
    simplicesᵢ::AbstractVector, degree::Integer, fatten::Real=def_fatten, 
    num_near_neigh::Integer=def_num_near_neigh, sigma::Real=0, 
    epm::Union{Nothing,EPM2D,EPM}=nothing,
    neighbor_method::NeighborMethod=def_neighbor_method, 
    num_neighbors::Union{Nothing,Integer}=nothing,
    weighted::Bool=false,constrained::Bool=true,atol::Real=def_atol)

    simplexᵢ = simplicesᵢ[index]
    simplex = Matrix(mesh.points[simplexᵢ,:]')
    dim = size(simplex,2)-1
    nsheets = size(eigenvals,1)
    if dim == 2
        nterms = ntripts(degree)
    else
        nterms = ntetpts(degree)
    end
    neighborsᵢ = reduce(vcat,[get_neighbors(s,ext_mesh,num_near_neigh) for s=simplexᵢ]) |> unique
    neighborsᵢ = filter(x -> !(x in simplexᵢ),neighborsᵢ)
    if num_neighbors == nothing
        num_neighbors = if dim == 2 def_num_neighbors2D else def_num_neighbors3D end
    end
    if length(neighborsᵢ) < num_neighbors num_neighbors = length(neighborsᵢ) end

    neighborsᵢ = select_neighbors(neighborsᵢ,simplex,ext_mesh,dim,num_neighbors,
        neighbor_method,epm)

    if neighbor_method == neighbors_inside
        n = def_inside_neighbors_divs # Number of points for the uniform sampling of the triangle
        b = sample_simplex(dim,n)
        b = b[:,setdiff(1:size(b,2),[1,n+1,size(b,2)])]
        eigvals = eval_epm(barytocart(b,simplex),epm)
    else
        b = carttobary(ext_mesh.points[neighborsᵢ,:]',simplex)
    end
    cornerind = simplex_cornerpts(dim,degree)
    if !constrained && neighbor_method != 3
        # Add corner points to b
        b = [sample_simplex(dim,degree)[:,cornerind] b]
    end
    M = Matrix(bernstein_basis(b,dim,degree)')
    if constrained && neighbor_method != 3
        # Remove terms at corners of basis function evaluations matrix
        ind = setdiff(1:nterms,cornerind)
        M = M[:,ind]
    end
 
    # Weighted least squares
    if weighted
        if dim == 2
            # Minimum distance from the edges of the triangle
            W = diagm([minimum([lineseg_pt_dist(ext_mesh.points[i,:],simplex[:,s]) for s=[[1,2],[2,3],[3,1]]])
                for i=neighborsᵢ])
        else
            # Minimum distance from the faces of the tetrahedron 
            W = diagm([minimum([ptface_mindist(ext_mesh.points[i,:],simplex[:,s]) for s=face_ind])
                for i=neighborsᵢ])
        end
    else
        W = I
    end
     
    # Only one sheet if calculating the coefficients of the sigma sheet.
    if sigma == 0
        bezcoeffs = [zeros(nterms) for i=1:nsheets]
        inter_bezcoeffs = [zeros(2,nterms) for i=1:nsheets]
    else
        bezcoeffs = [zeros(nterms) for i=1:1]
        inter_bezcoeffs = [zeros(2,nterms) for i=1:1]
    end
     
    for sheet = 1:nsheets
        if sigma == 0
            if neighbor_method != 3
                fᵢ = eigenvals[sheet,sym_unique[neighborsᵢ]]
            else
                fᵢ = eigvals[sheet,:]
            end
            q = eigenvals[sheet,sym_unique[simplexᵢ]]
        else
            if neighbor_method != 3
                fᵢ = [sum(eigenvals[1:sigma,sym_unique[neighborsᵢ]],dims=1)...]
            else
                fᵢ = [sum(eigvals[1:sigma,:],dims=1)...]
            end
            q = [sum(eigenvals[1:sigma,sym_unique[simplexᵢ]],dims=1)...]
        end

        if constrained
            Z = fᵢ - (b.^2)'*q
        else
            fᵢ = [q; fᵢ];
            Z = fᵢ
        end
     
        if weighted
            MWM = M'*W*M
            if isapprox(det(MWM),0,atol=atol)
                c = pinv(MWM)*M'*W*Z
            else
                c = inv(MWM)*M'*W*Z
            end
        else
            # c = pinv(M)*Z        
            c = M\Z
        end
        if constrained
            scoeffs = zeros(nterms)
            scoeffs[cornerind] = q
            scoeffs[setdiff(1:nterms,cornerind)] = c
        else
            scoeffs = c
        end
        bezcoeffs[sheet] = scoeffs
        qᵢ = [eval_poly(b[:,i],scoeffs,dim,degree) for i=1:size(b,2)]
        δᵢ = fᵢ - qᵢ; 
        ϵ = Matrix(reduce(hcat,[(1/dot(M[i,:],M[i,:])*δᵢ[i])*M[i,:] for i=1:length(δᵢ)])')
        ϵ = [minimum(ϵ,dims=1); maximum(ϵ,dims=1)].*fatten
        intercoeffs = [c';c'] .+ ϵ
        if constrained
            ic = zeros(2,nterms)
            ic[:,cornerind] = [q q]'
            ic[:,setdiff(1:nterms,cornerind)] = intercoeffs
            intercoeffs = ic
        end

        inter_bezcoeffs[sheet] = intercoeffs
        if sigma != 0
            break
        end
    end

    Vector{Matrix{Float64}}(inter_bezcoeffs),Vector{Vector{Float64}}(bezcoeffs)
end

@doc """
    init_exactfit(epm; num_kpoints,polydegree,atol,rtol)

Calculate the polynomial coefficients for an exact fit, among other quantities.

# Arguments
- `epm::Union{EPM,EPM2D}`: an empirical pseudopotential object.
- `num_kpoints::Integer`: the number of k-points in the mesh.
- `polydegree::Int`: the degree of the polynomial.

# Returns
- `unique_pts::Matrix{<:Real}`: a matrix with the unique k-points as columns (but not rotationally unique).
- `mesh::PyObject`: a triangulation or tetrahedralization of the IBZ from QHull called
    from SciPy.
- `simplicesᵢ::Vector`: the indices of points at the corners of the simplices in `mesh`.
- `sym_unique::Vector{<:Integer}`: the indices of rotationally unique points in `unique_pts`.
- `eigenvals::Matrix{<:Real}`: the eigenvalues at the rotationally unique points.
- `mesh_bezcoeffs::Vector`: the coefficients of the polynomial for each tile and sheet.
- `mesh_intcoeffs::Vector`: the interval coefficients (length 0 in the case of exact fit).

# Examples
```
using Pebsi.EPMs, Pebsi.QuadraticIntegration
init_exactfit(m11,init_msize=3,polydegree=1)
```
"""
function init_exactfit(epm::Union{EPM,EPM2D}; num_kpoints::Integer, polydegree::Integer,
    atol::Real=def_atol,rtol::Real=def_rtol)
    dim = size(epm.recip_latvecs,1)
    mesh,simplicesᵢ = ibz_initmesh(epm.ibz,num_kpoints)
    simplices = [Array(mesh.points[s,:]') for s=simplicesᵢ]
    bspts = sample_simplex(dim,polydegree);
    pts = [barytocart(bspts,s) for s=simplices]
    nptfun = if dim == 2 ntripts else ntetpts end
    ptsᵢ = [zeros(Int,nptfun(polydegree)) for i=1:length(simplices)]
    npad = 2^dim
    unique_pts = [mesh.points[1:npad,:]' zeros(size(reduce(hcat,pts)))]
    npolypts = nptfun(polydegree)
    n = npad+1; ptsᵢ[1][1] = n; unique_pts[:,n] = pts[1][:,1]
    for s=1:length(simplicesᵢ)
        for p = 1:npolypts
            utest = vec(mapslices(x->isapprox(pts[s][:,p],x,atol=atol,rtol=rtol),unique_pts[:,1:n],dims=1))
            pos = findall(x->x==1,utest)
            if pos == []
                n += 1
                ptsᵢ[s][p] = n
                unique_pts[:,n] = pts[s][:,p]
            else
                ptsᵢ[s][p] = pos[1]
            end
        end
    end
    unique_pts = unique_pts[:,1:n]
    cvpts = get_cvpts(unique_pts,epm.ibz);
    # Points on the boundary may be symmetrically unique
    sym_unique,unique_pts = get_sym_unique!(unique_pts,epm.pointgroup,cvpts=cvpts)
    ptsᵢ = [sym_unique[p] for p=ptsᵢ]
    uniqueᵢ = sort(unique(sym_unique))[2:end]
    eigenvals = zeros(epm.sheets,length(uniqueᵢ)+npad)
    for i=uniqueᵢ
        eigenvals[:,i] = eval_epm(unique_pts[:,i], epm, rtol=rtol, atol=atol)
    end
    mesh_bezcoeffs = [[getpoly_coeffs(eigenvals[j,ptsᵢ[i]],bspts,dim,polydegree) 
            for j=1:epm.sheets] for i=1:length(simplicesᵢ)]

    # Adjustments to coefficients so that the semi-analytic quadratic integration
    # methods can be used when the degree of the polynomial is 0 or 1.
    if polydegree == 0
        mesh_bezcoeffs = [[[v[1] for i=1:nptfun(2)] for v=c] for c=mesh_bezcoeffs]
    elseif polydegree == 1
        if dim == 2
        mesh_bezcoeffs = [[[v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,(v[2]+v[3])/2,v[3]] for v=c] 
            for c=mesh_bezcoeffs]
        else
            mesh_bezcoeffs = [[[v[1],(v[1]+v[2])/2,v[2],(v[1]+v[3])/2,(v[2]+v[3])/2,
                v[3],(v[1]+v[4])/2,(v[2]+v[4])/2,(v[3]+v[4])/2,v[4]] for v=c] 
                for c=mesh_bezcoeffs]
        end
    end
    mesh_intcoeffs = [[Matrix([mesh_bezcoeffs[i][j] mesh_bezcoeffs[i][j]]') 
        for j=1:epm.sheets] for i=1:length(simplices)]
    (unique_pts, mesh, simplicesᵢ, sym_unique, eigenvals, mesh_bezcoeffs, mesh_intcoeffs)
end
