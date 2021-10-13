module Plotting

using PyPlot: subplots, figure, PyObject, figaspect, plt, pyimport

using QHull: chull

using ..QuadraticIntegration: bandstructure
using ..EPMs: epm₋model2D, epm₋model
using ..RectangularMethod: sample_unitcell
using ..Polynomials: eval_poly, sample_simplex ,eval_bezcurve
using ..Geometry: carttobary, barytocart, simplex_size

using SymmetryReduceBZ.Plotting: plot_2Dconvexhull
using SymmetryReduceBZ.Utilities: sortpts2D

using Statistics: mean

@doc """
    meshplot(meshpts,ax,color)

Plot the points within a mesh in 2D or 3D.
"""
function meshplot(meshpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; color::String="blue", alpha=1.,
    zorder::Int=0)

    dim = size(meshpts,1)
    if dim == 2
        if ax == nothing
            (fig,ax) = subplots()
        end
        ϵ=0.1*max(meshpts...)

        (xs,ys)=[meshpts[i,:] for i=1:dim]
        ax.scatter(xs,ys,color=color,alpha=alpha,zorder=zorder)
        ax.set_aspect(1)

    elseif dim == 3
        if ax == nothing
            fig = figure()
            ax = fig.add_subplot(111, projection="3d")
        end
        (xs,ys,zs)=[meshpts[i,:] for i=1:dim]
        ax.scatter3D(xs,ys,zs,color=color,alpha=alpha,zorder=zorder)

        ϵ=0.1*max(meshpts...)
        plotrange=[[minimum(meshpts[:,i])-ϵ,
            maximum(meshpts[:,i])+ϵ] for i=1:3]
        ax.auto_scale_xyz(plotrange[1],plotrange[2],plotrange[3])

    else
        raise(ArgumentError("The meshpoints must be in an arrray with 2 or 3
            rows. Each mesh point is a column of the array in Cartesian
            coordinates."))
    end
    
    ax
end

@doc """
    contourplot(bezpts,ax;filled=false,padded=true)

Plot the level curves of a polynomial surface.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier point of the quadratic surface
    as columns of an array.
- `ax::Union{PyObject,Nothing}`: an axes object.
- `filled::Bool=false`: if true, the regions below isovalue are shadded.
- `padded::Bool=true`: if true, the region around the triangle is also plotted.

# Returns
- `ax::PyObject`: the plot axes object.

"""
function contourplot(bezpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; filled::Bool=false,
    padded::Bool=true,ndiv::Integer=100,colors=["black","white"],
    alpha::Real=0.5,curvewidths=1,zorder::Int=1)::PyObject
    dim = 2
    deg = 2
    # Plot triangle
    coeffs = bezpts[3,:]
    simplex = bezpts[1:2,[1,3,6]]
    if ax == nothing; (fig,ax)=subplots() end
    if padded
        N = [ndiv 0; 0 ndiv]
        basis = [simplex[:,2] - simplex[:,1] simplex[:,3] - simplex[:,1]]
        grid_offset = [0.5,0.5]
        plotpts = 2 .* sample_unitcell(basis,N,grid_offset)
        plotpts = mapslices(x->x-basis*[1/2,1/2]+simplex[:,1],plotpts,dims=1)
        bplotpts = carttobary(plotpts,simplex)
        plotvals= eval_poly(bplotpts,coeffs,dim,deg)
        X=reshape(plotpts[1,:],(ndiv,ndiv))
        Y=reshape(plotpts[2,:],(ndiv,ndiv))
        Z=reshape(plotvals,(ndiv,ndiv))

        if filled 
            ax.contourf(X,Y,Z,[-1e10,0],colors=colors,alpha=alpha,
                linewidths=curvewidths,zorder=zorder)
        else
            ax.contour(X,Y,Z,[0],colors=colors[1],linewidths=curvewidths,
                zorder=zorder)
        end
    else
        bpts = sample_simplex(2,100)
        pts = barytocart(bpts,simplex)
        vals = eval_poly(bpts,coeffs,dim,deg)
        if filled
            ax.tricontourf(pts[1,:],pts[2,:],vals,[-1e19,0],colors=colors,
                linewidths=curvewidths,alpha=alpha,zorder=zorder)
        else
            ax.tricontour(pts[1,:],pts[2,:],vals,[0],[0],colors=colors[1],
                linewidths=curvewidths,zorder=zorder)
        end
    end
    ax
end

function contourplot(ebs::bandstructure, ax::Union{PyObject,Nothing}=nothing;
    sort=false, linewidth::Real=0.5, edgecolor::String="black", 
    filled::Bool=false, ndiv::Integer=100, colors=["black","white"], 
    alpha_curve::Real=1, curvewidths::Real=0.5)::PyObject 
     
    patch=pyimport("matplotlib.patches")
    collections=pyimport("matplotlib.collections")

    bpts = sample_simplex(2,2)
    tripts = [Array(ebs.mesh.points[i,:]') for i=ebs.simplicesᵢ]

    if ax == nothing fig,ax=subplots() end

    # Locate triangles and sheets that are partially occupied
    pocc = [findall(x->x==1,b) for b=ebs.partially_occupied]
    indices = reduce(vcat,
        filter(x->!(x==nothing),
            [pocc[i] == [] ? nothing : [[i,j] for j=pocc[i]] for i=1:length(pocc)]))

    sizes = [simplex_size(t) for t=tripts]
    ρ = ndiv/maximum(sizes)
    ndivs = round.(Int,ρ*sizes)
    
    contour_colors = ["blue","red","blue"]
    fls = [ebs.fermilevel_interval[2],ebs.fermilevel,ebs.fermilevel_interval[1]]

    for (j,cfun) in enumerate([minimum,mean,maximum])
        for i=indices
            bezpts = [barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[i[1]],:]'); 
                cfun(ebs.mesh_intcoeffs[i[1]][i[2]],dims=1) .- fls[j]]
                
        
            ax = contourplot(bezpts,ax,padded=false,ndiv=ndivs[i[1]],
                alpha=alpha_curve,curvewidths=curvewidths,colors=[contour_colors[j]],
                filled=filled,zorder=2)
        end
    end

    # Plot the triangles.
    # Values for the color of the filling of triangles.
    bandenergy_errors = abs.(ebs.bandenergy_errors)
    bandenergy_errors = map(x->x < 1e-15 ? 1e-15 : x,bandenergy_errors)

    cm = plt.get_cmap("binary")
    bmin = minimum(bandenergy_errors)
    bmax = maximum(bandenergy_errors)
    vmin = 0
    vmax = 200
    tcolors = round.(Int,map(x->(log(x)-log(bmin))/abs(log(bmax) - log(bmin))*vmax,bandenergy_errors))

    ibz_volume = sum(sizes)
    err_cutoff = [simplex_size(s)/ibz_volume for s=tripts]*ebs.target_accuracy;
    err_ratios = round.(Int,bandenergy_errors ./ err_cutoff)
    @show minimum(err_ratios)
    # tcolors = map(x -> log(x)/log(vmax)*vmax, err_ratios)

    for (i,t) in enumerate(tripts)
        ax = polygonplot(t,ax,facecolor=cm(tcolors[i]),
            linewidth=linewidth,edgecolor="gray",zorder=1)
    end

    # Set plot range.
    xrange = [minimum(ebs.mesh.points[5:end,1]), maximum(ebs.mesh.points[5:end,1])]
    xrange = [xrange[1] - 0.05*diff(xrange)[1], xrange[2] + 0.05*diff(xrange)[1]]
    yrange = [minimum(ebs.mesh.points[5:end,2]), maximum(ebs.mesh.points[5:end,2])]
    yrange = [yrange[1] - 0.05*diff(yrange)[1], yrange[2] + 0.05*diff(yrange)[1]]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)        
    ax
end

"""
    bezplot(bezpts)

Plot a quadratic Bezier curve and its Bezier points (1D).
"""
function bezplot(bezpts::AbstractMatrix{<:Real},
        ax::Union{PyObject,Nothing}=nothing; zorder=zorder)
    
    dim = 1
    deg = 2
    simplex = [bezpts[1,1] bezpts[1,end]]
    s = 0.1*abs(simplex[2]-simplex[1])
    sgn = sign(simplex[2]-simplex[1])
    pts = Array(collect(simplex[1]-s:sgn*0.01:simplex[2]+s)')
    bpts = carttobary(pts,simplex)
    vals = eval_poly(bpts,bezpts[end,:],dim,deg)
    fvals = zeros(length(pts))
    if ax == nothing
        (fig,ax)=subplots()
    end
    ax.plot(pts[:],vals,pts[:],fvals)
    ax.plot(bezpts[1,:],bezpts[2,:],"bo")
    ax
end

"""
    bezcurve_plot(bezptsᵣ,bezwtsᵣ,ax)

Plot a rational quadratic Bezier curve and its Bezier points

# Arguments
- `bezptsᵣ::AbstractMatrix{<:Real}`: the Bezier control points as columns of an
    array in Cartesian coordinates.
- `bezwtsᵣ::AbstractVector{<:Real}`: the weights of the control points.
- `ax::Union{PyObject,Nothing}=nothing`: the figure's axes.

# Returns
- `ax::PyObject`: the axes of the figure.

# Examples
```jldoctest
bezpts = [0.0 0.0 1.0; 1.0 1/3 0.0]
bezwts = [1.0, 1.5, 1.0]
bezcurve_plot(bezpts,bezwts)
# output
PyObject <AxesSubplot:>
```
"""
function bezcurve_plot(bezptsᵣ::AbstractMatrix{<:Real},
    bezwtsᵣ::AbstractVector{<:Real},
    ax::Union{PyObject,Nothing}=nothing;zorder::Int=0)::PyObject
    if ax == nothing; (fig,ax)=subplots(); end
    
    data = eval_bezcurve(collect(0:1/1000:1),bezptsᵣ,bezwtsᵣ)
    ax = meshplot(bezptsᵣ,ax)
    ax.plot(data[1,:],data[2,:],color="blue",zorder=zorder)
    ax
end

@doc """
    polygonplot()

Make a plot of a polygon.

# Arguments
- `pts::Matrix{<:Real}`: cartesian coordinates of a convex polygon as columns of an array.
- `ax::PyObject`: an axes object from matplotlib.
- `facecolor::String="blue"`: the color of the area within the convex hull.
- `alpha::Real=0.3`: the transparency of the convex hull.
- `linewidth::Real=3`: the width of the edges.
- `edgecolor::String="black"`: the color of the edges.

# Returns
- `ax::PyObject`: updated `ax` that includes a plot of the convex polygon.
```
"""
function polygonplot(pts::Matrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; sort=false, facecolor="None",
    alpha::Real=1.0, linewidth::Real=0.5, edgecolor="black",
    zorder::Int=0)::PyObject

    if sort
        perm = sortpts2D(pts)
        pts = pts[:,perm]
    end
    if ax == nothing 
        println("not ok")
        fig,ax = subplots()
    end

    # ax.add_patch(plt.Polygon(Array(pts'),edgecolor=edgecolor,facecolor=facecolor,
    #     alpha=alpha,linewidth=linewidth))
    ax.fill(pts[1,:],pts[2,:],edgecolor=edgecolor,facecolor=facecolor,
        alpha=alpha,linewidth=linewidth,zorder=zorder)
    ax
end

function polygonplot(polygons::Vector{Matrix{T}} where T<:Real,
    ax::Union{PyObject,Nothing}=nothing;sort=false, facecolor="None",
    alpha::Real=1.0, linewidth::Real=0.5, edgecolor="black", zorder::Int=0)
    if ax == nothing fig,ax = subplots() end

    for poly in polygons
        ax = polygonplot(poly,ax;sort=sort,facecolor=facecolor,alpha=alpha,
            linewidth=linewidth,edgecolor=edgecolor,zorder=zorder)
    end
    ax
end

@doc """
    plot_bandstructure(name,basis,rules,expansion_size,sheets,kpoint_dist,
        convention,coordinates)

Plot the band structure of an empirical pseudopotential.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `kpoint_dist::Real`: the distance between k-points in the band plot.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::Int`: the sheets included in the electronic band structure plot.

# Returns
- (`fig::PyPlot.Figure`,`ax::PyCall.PyObject`): the band structure plot
    as a `PyPlot.Figure`.
"""
function plot_bandstructure(epm::Union{epm₋model2D,epm₋model},
    kpoint_dist::Real,expansion_size::Integer;
    func::Union{Nothing,Function}=nothing,sheets::Integer=10)

    sp=pyimport("seekpath")

    basis = [epm.real_latvecs[:,i] for i=1:size(epm.real_latvecs,1)]
    # basis = epm.real_latvecs
    rbasis=epm.recip_latvecs
    atomtypes=epm.atom_types
    atompos=[[0,0,0]]

    # Calculate the energy cutoff of Fourier expansion.
    cutoff=1
    num_terms=0
    rtol=0.2
    atol=10
    while (abs(num_terms - expansion_size) > expansion_size*rtol &&
        abs(num_terms - expansion_size) > atol)
        if num_terms - expansion_size > 0
            cutoff *= 0.95
        else
            cutoff *= 1.1
        end
        num_terms = size(sample_sphere(rbasis,cutoff,[0,0,0]),2)
    end
    
    # Calculate points along symmetry paths using `seekpath` Python package.
    # Currently uses high symmetry paths from the paper: Y. Hinuma, G. Pizzi,
    # Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on
    # crystallography, Comp. Mat. Sci. 128, 140 (2017).
    # DOI: 10.1016/j.commatsci.2016.10.015
    structure=[basis,atompos,atomtypes]
    timereversal=true

    spdict=sp[:get_explicit_k_path](structure,timereversal,kpoint_dist)
    sympath_pts=Array(spdict["explicit_kpoints_abs"]')

    if coordinates == "lattice"
        m=spdict["reciprocal_primitive_lattice"]
        sympath_pts=inv(m)*sympath_pts
    elseif convention == "ordinary"
        sympath_pts=1/(2π).*sympath_pts
    end

    # Determine the x-axis tick positions and labels.
    labels=spdict["explicit_kpoints_labels"]
    sympts_pos = filter(x->x>0,[if labels[i]==""; -1 else i end for i=1:length(labels)])
    λ=spdict["explicit_kpoints_linearcoord"]

    tmp_labels=[labels_dict[l] for l=labels[sympts_pos]]
    tick_labels=tmp_labels
    for i=2:(length(tmp_labels)-1)
        if (sympts_pos[i-1]+1) == sympts_pos[i]
            tick_labels[i]=""
        elseif (sympts_pos[i]+1) == sympts_pos[i+1]
            tick_labels[i]=tmp_labels[i]*"|"*tmp_labels[i+1]
        else
            tick_labels[i]=tmp_labels[i]
        end
    end

    # Eigenvalues in band structure plot
    evals = eval_epm(sympath_pts,epm,sheets=sheets)

    fig,ax=subplots()
    for i=1:epm.sheets ax.scatter(λ,evals[i,:],s=0.1) end
    ax.set_xticklabels(tick_labels)
    ax.set_xticks(λ[sympts_pos])
    ax.grid(axis="x",linestyle="dashed")
    ax.set_xlabel("High symmetry points")
    ax.set_ylabel("Total energy (eV)")
    ax.set_title(epm.name*" band structure plot")
    (fig,ax)
end

@doc """
    plot_bandstructure(name,basis,rules,expansion_size,sheets,kpoint_dist,
        convention,coordinates)
Plot the band structure of an empirical pseudopotential.
# Arguments
- `name`::String: the name of metal.
- `basis::AbstractMatrix{<:Real}`: the lattice vectors of the crystal
    as columns of a 3x3 array.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::Int`: the sheets included in the electronic
    band structure plot.
- `kpoint_dist::Real`: the distance between k-points in the band plot.
- `convention::String="angular"`: the convention for going from real to
    reciprocal space. Options include "angular" and "ordinary".
- `coordinates::String="Cartesian"`: the coordinates of the k-points in
    the band structure plot. Options include "Cartesian" and "lattice".
# Returns
- (`fig::PyPlot.Figure`,`ax::PyCall.PyObject`): the band structure plot
    as a `PyPlot.Figure`.
# Examples
```jldoctest
import Pebsi.EPMs: eval_epm,plot_bandstructure
name="Al"
Al_latvecs=[0.0 3.8262 3.8262; 3.8262 0.0 3.8262; 3.8262 3.8262 0.0]
Al_rules=Dict(2.84 => 0.0562,1.42 => 0.0179)
cutoff=100
sheets=10
kpoint_dist=0.001
plot_bandstructure(name,Al_latvecs,Al_rules,cutoff,sheets,kpoint_dist)
# returns
(PyPlot.Figure(PyObject <Figure size 1280x960 with 1 Axes>),
PyObject <AxesSubplot:title={'center':'Al band structure plot'},
xlabel='High symmetry points', ylabel='Total energy (Ry)'>)
"""
function plot_bandstructure(name::String,basis::AbstractMatrix{<:Real},
        rules::Dict{<:Real,<:Real},expansion_size::Integer,
        sheets::Int,kpoint_dist::Real,
        convention::String="angular",coordinates::String="Cartesian";
        func::Union{Nothing,Function}=nothing)

    sp=pyimport("seekpath")

    rbasis=get_recip_latvecs(basis,convention)
    atomtypes=[0]
    atompos=[[0,0,0]]

    # Calculate the energy cutoff of Fourier expansion.
    cutoff=1
    num_terms=0
    tol=0.2
    while abs(num_terms - expansion_size) > expansion_size*tol
        if num_terms - expansion_size > 0
            cutoff *= 0.95
        else
            cutoff *= 1.1
        end
        num_terms = size(sample_sphere(rbasis,cutoff,[0,0,0]),2)
    end

    # Calculate points along symmetry paths using `seekpath` Python package.
    # Currently uses high symmetry paths from the paper: Y. Hinuma, G. Pizzi,
    # Y. Kumagai, F. Oba, I. Tanaka, Band structure diagram paths based on
    # crystallography, Comp. Mat. Sci. 128, 140 (2017).
    # DOI: 10.1016/j.commatsci.2016.10.015
    structure=[basis,atompos,atomtypes]
    timereversal=true

    spdict=sp[:get_explicit_k_path](structure,timereversal,kpoint_dist)
    sympath_pts=Array(spdict["explicit_kpoints_abs"]')

    if coordinates == "lattice"
        m=spdict["reciprocal_primitive_lattice"]
        sympath_pts=inv(m)*sympath_pts
    elseif convention == "ordinary"
        sympath_pts=1/(2π).*sympath_pts
    end

    # Determine the x-axis tick positions and labels.
    labels=spdict["explicit_kpoints_labels"];
    sympts_pos = filter(x->x>0,[if labels[i]==""; -1 else i end for i=1:length(labels)])
    λ=spdict["explicit_kpoints_linearcoord"];

    tmp_labels=[labels_dict[l] for l=labels[sympts_pos]]
    tick_labels=tmp_labels
    for i=2:(length(tmp_labels)-1)
        if (sympts_pos[i-1]+1) == sympts_pos[i]
            tick_labels[i]=""
        elseif (sympts_pos[i]+1) == sympts_pos[i+1]
            tick_labels[i]=tmp_labels[i]*"|"*tmp_labels[i+1]
        else
            tick_labels[i]=tmp_labels[i]
        end
    end

    # Eigenvalues in band structure plot
    evals = eval_epm(sympath_pts,rbasis,rules,cutoff,sheets,func=func)

    fig,ax=subplots()
    for i=1:10 ax.scatter(λ,evals[i,:],s=0.1) end
    ax.set_xticklabels(tick_labels)
    ax.set_xticks(λ[sympts_pos])
    ax.grid(axis="x",linestyle="dashed")
    ax.set_xlabel("High symmetry points")
    ax.set_ylabel("Total energy (Ry)")
    ax.set_title(name*" band structure plot")
    (fig,ax)
end

end # module
