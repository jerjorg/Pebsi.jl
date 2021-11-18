module Plotting

using PyPlot: subplots, figure, PyObject, plt, pyimport
using Statistics: mean

using ..QuadraticIntegration: bandstructure
using ..EPMs: epm₋model2D, epm₋model, eval_epm
using ..RectangularMethod: sample_unitcell
using ..Polynomials: eval_poly, sample_simplex, eval_bezcurve
using ..Geometry: carttobary, barytocart, simplex_size
using ..Mesh: notbox_simplices

using SymmetryReduceBZ.Utilities: sortpts2D, sample_sphere
using SymmetryReduceBZ.Lattices: get_recip_latvecs

export meshplot, contourplot, bezplot, bezcurve_plot, polygonplot, 
    plot_bandstructure

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]    

@doc """
    meshplot(meshpts,ax,color)

Plot the points in a mesh in 2D or 3D.

# Arguments
- `meshpts::AbstractMatrix{<:Real}`: the points in the mesh in columns of a
    matrix in Cartesian coordinates.
- `ax::Union{PyObject,Nothing}=nothing`: a `PyPlot` axes object.
- `color::String="blue"`: the fill color of the points in the plot.
- `alpha::Real=1.`: the transparency of the points.
- `zorder::Integer=0`: a parameter that specifies the layer of the plot when
    overlaying multiple plots.

# Returns
- `ax::PyObject`: an axes object from `PyPlot`.

# Examples
```jldoctest
using PyPlot: subplots
using Pebsi.Plotting: meshplot
fig,ax = subplots()
mesh = [0 1 1 0; 0 0 1 1]
ax = meshplot(mesh,ax)
# output
PyObject <AxesSubplot:>
```

```jldoctest
using PyPlot: figure, subplot
using Pebsi.Plotting: meshplot
fig = figure()
ax = subplot(projection="3d")
mesh = [0 1 1 0 0 1 1 0; 0 0 1 1 0 0 1 1; 0 0 0 0 1 1 1 1]
ax = meshplot(mesh,ax)
# output
PyObject <Axes3DSubplot:>
```
"""
function meshplot(meshpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; color::String="blue", alpha::Real=1.,
    zorder::Integer=0)::PyObject

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
        plotrange=[[minimum(meshpts[i,:])-ϵ,
            maximum(meshpts[i,:])+ϵ] for i=1:3]
        ax.auto_scale_xyz(plotrange[1],plotrange[2],plotrange[3])

    else
        raise(ArgumentError("The meshpoints must be in an arrray with 2 or 3
            rows. Each mesh point is a column of the array in Cartesian
            coordinates."))
    end 
    ax
end

@doc """
    meshplot(mesh::PyObject,ax,...)

Plot the triangles in a triangular mesh.

# Arguments
- `mesh::PyObject`: a mesh object obtained through `PyCall` from 
    `scipy.spatial.Delaunay`.
- `ax::Union{PyObject,Nothing}=nothing`: an axes object from `PyPlot`. 
- `facecolor::String="None"`: the color of the interior of the triangles.
- `alpha::Real=1.`: the transparency of the triangles.
- `linewidth::Real=0.5`: the width of the edges of the triangles.
- `edgecolor::String="black"`: the color of the edges of the triangles.
- `zorder::Integer=0`: the layer of the plot when overlaying multiple plots.

# Returns
- `ax::PyObject`: an axes object from `PyPlot`.

# Examples
```jldoctest
using Pebsi.Mesh: ibz_init₋mesh
using Pebsi.EPMs: m11
using Pebsi.Plotting: meshplot
mesh = ibz_init₋mesh(m11.ibz,3)
fig,ax = subplots()
ax = meshplot(mesh,ax)
# output
PyObject <AxesSubplot:>
```
"""
function meshplot(mesh::PyObject,ax::Union{PyObject,Nothing}=nothing;
    facecolor::String="None",alpha::Real=1.,linewidth::Real=0.5,
    edgecolor::String="black",zorder::Integer=0)::PyObject    
     
    simplicesᵢ = notbox_simplices(mesh)
    simplices = [Array(mesh.points[s,:]') for s=simplicesᵢ]
    if ax == nothing fig,ax = subplots() end
    ax = polygonplot(simplices,ax; facecolor=facecolor,alpha=alpha,
        linewidth=linewidth,edgecolor=edgecolor,zorder=zorder)
    ax
end

@doc """
    contourplot(bezpts,ax;filled=false,padded=true)

Plot the level curves of a polynomial surface.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier point of the quadratic surface
    as columns of an array.
- `ax::Union{PyObject,Nothing}`: an axes object.
- `filled::Bool=false`: if true, the regions below 0 are shadded.
- `padded::Bool=true`: if true, the region around the triangle is also plotted.
- `ndiv::Integer=100`: the number of devisions along the edge of the triangle
    for the plot. Essentially determines the density of the grid.
- `colors::AbstractVector{String}=["black","white"]`: the colors of shaded regions
    above and below zero in the contour plots. The first color is the color of
    level curves when not shading regions.
- `alpha::Real=0.5`: the transparency of the plot.
- `linewidths::Real=1`: the widths of the contour lines.
- `zorder::Integer=1`: the order of the plot layer when overlaying multiple plots.

# Returns
- `ax::PyObject`: the plot axes object.

# Examples
```jldoctest
using PyPlot: subplots
using Pebsi.Plotting: contourplot
fig,ax = subplots()
bezpts = [0. 0.5 1. 0.5 1. 1.; 0. 0. 0. 0.5 0.5 1.; -1 0.1 -0.2 1.2 1.1 -0.2]
ax = contourplot(bezpts,ax,padded=false,filled=true,ndiv=200,linewidths=2,
    colors=["red","black"],zorder=2,alpha=0.5)
# output
PyObject <AxesSubplot:>
```
"""
function contourplot(bezpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; filled::Bool=false,
    padded::Bool=true,ndiv::Integer=100,colors::AbstractVector{String}=["black","white"],
    alpha::Real=0.5,linewidths::Real=1,zorder::Integer=1)::PyObject
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
                linewidths=linewidths,zorder=zorder)
        else
            ax.contour(X,Y,Z,[0],colors=colors[1],linewidths=linewidths,
                zorder=zorder)
        end
    else
        bpts = sample_simplex(2,100)
        pts = barytocart(bpts,simplex)
        vals = eval_poly(bpts,coeffs,dim,deg)
        if filled
            ax.tricontourf(pts[1,:],pts[2,:],vals,[-1e19,0],colors=colors,
                linewidths=linewidths,alpha=alpha,zorder=zorder)
        else
            ax.tricontour(pts[1,:],pts[2,:],vals,[0],[0],colors=colors[1],
                linewidths=linewidths,zorder=zorder)
        end
    end
    ax
end

@doc """
    contourplot(ebs,ax;sort,linewidth,edgecolor,filled,ndiv,colors,alpha_curve,
        curvewidths)

Plot the triangles shaded by error and approx. Fermi curve of an approx. of the band structure.

# Arguments
- `ebs::bandstructure`: a container for the band structure approximation.
- `ax::Union{PyObject,Nothing}=nothing`: an axes object from `PyPlot`.
- `sort::Bool=false`: if true, sort the vertices of the 
- `linewidth::Real=0.5`: widths of edges of triangles  
- `edgecolor::String="black"`: the colors of triangle edges
- `filled::Bool=false`: if true, color the regions of the band structure below 0.
- `ndiv::Integer=100`: the number of divisions of the largest triangles. Divisions
    of smaller triangles are scaled down.
- `colors::AbstractVector{String}=["black","white"]`: the color of filled regions.
- `alpha_curve::Real=1`: the transparency of the level curve.
- `curvewidths::Real=0.5`: the width of the level curves.

# Returns
- `ax::PyObject`: a `PyPlot` axes object. 

# Examples
```jldoctest
using Pebsi.EPMs: m11
using Pebsi.QuadraticIntegration: init_bandstructure, calc_flbe!
using Pebsi.Plotting: contourplot
epm = m11
ebs = init_bandstructure(epm)
calc_flbe!(epm,ebs)
ax = contourplot(ebs)
# output
PyObject <AxesSubplot:>
```
"""
function contourplot(ebs::bandstructure, ax::Union{PyObject,Nothing}=nothing;
    sort::Bool=false, linewidth::Real=0.5, edgecolor::String="black", 
    filled::Bool=false, ndiv::Integer=100,
    colors::AbstractVector{String}=["black","white"], alpha_curve::Real=1, 
    curvewidths::Real=0.5)::PyObject 
     
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
    # fls = [ebs.fermilevel,ebs.fermilevel,ebs.fermilevel]

    for (j,cfun) in enumerate([minimum,mean,maximum])
        for i=indices
            bezpts = [barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[i[1]],:]'); 
                cfun(ebs.mesh_intcoeffs[i[1]][i[2]],dims=1) .- fls[j]]
                
        
            ax = contourplot(bezpts,ax,padded=false,ndiv=ndivs[i[1]],
                alpha=alpha_curve,linewidths=curvewidths,colors=[contour_colors[j]],
                filled=filled,zorder=2)
        end
    end

    # Plot the triangles.
    # Values for the color of the interion of triangles.
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

    for (i,t) in enumerate(tripts)
        ax = polygonplot(t,ax,facecolor=cm(tcolors[i]),
            linewidth=linewidth,edgecolor="gray",sort=sort,zorder=1)
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

Plot a univariate quadratic Bezier curve and its Bezier points.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadrtic curve as
    columns of a matrix.
- `ax::Union{PyObject,Nothing}=nothing`:  an axes object from `PyPlot`.
- `zorder::Int=1`: the layer of the plot when overlaying multiple plots. 

# Returns
- `ax::PyObject`: an axes object

# Examples
```jldoctest
using PyPlot: subplots
using Pebsi.Plotting: bezplot
fig,ax = subplots()
bezpts = [0. 0.5 1.; -0.2 1.3 -0.5]
ax = bezplot(bezpts,ax)
# output
PyObject <AxesSubplot:>
```
"""
function bezplot(bezpts::AbstractMatrix{<:Real},
        ax::Union{PyObject,Nothing}=nothing; zorder::Int=1)
    dim = 1; deg = 2
    simplex = [bezpts[1,1] bezpts[1,end]]
    l = abs(simplex[2]-simplex[1]); s = l/10.
    sgn = sign(simplex[2]-simplex[1])
    pts = Array(collect(simplex[1]-s*sgn:sgn*l/100:simplex[2]+s*sgn)')
    bpts = carttobary(pts,simplex)
    vals = eval_poly(bpts,bezpts[end,:],dim,deg)
    fvals = zeros(length(pts))
    if ax == nothing
        (fig,ax)=subplots()
    end
    ax.plot(pts[:],vals,pts[:],fvals)

    ax.plot(bezpts[1,:],bezpts[end,:],"bo")
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
```
using Pebsi.Plotting: bezcurve_plot
bezpts = [0.0 0.0 1.0; 1.0 1/3 0.0]
bezwts = [1.0, 1.5, 1.0]
ax = bezcurve_plot(bezpts,bezwts)
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
    polygonplot(pts,ax)

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
    zorder::Int=0,label::String="_nolegend_")::PyObject

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
        alpha=alpha,linewidth=linewidth,zorder=zorder,label=label)
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
- `name`::String: the name of metal.
- `basis::AbstractMatrix{<:Real}`: the lattice vectors of the crystal
    as columns of a 3x3 array.
- `atomtypes::Vector{<:Integer}`: an integer atom label for all atoms
- `atompos::Matrix{<:Integer}`: the positions of the atoms in columns of a 
    matrix.
- `rules::Dict{Float64,Float64}`: a dictionary whose keys are distances between
    reciprocal lattice points rounded to two decimals places and whose values
    are the empirical pseudopotential form factors.
- `ax::Union{PyObject,Nothing}`: an axes object from PyPlot on which to plot the
    band structure.
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
function plot_bandstructure(name::String, basis::AbstractMatrix{<:Real},
    atomtypes::Vector{<:Integer}, atompos::Matrix{<:Real},
    rules, ax::Union{PyObject,Nothing}=nothing; 
    expansion_size::Integer=1000, sheets::Int=10, kpoint_dist::Real=0.01, 
    convention::String="angular", coordinates::String="Cartesian",
    func::Union{Nothing,Function}=nothing)

    sp=pyimport("seekpath")
    rbasis=get_recip_latvecs(basis,convention)
    atompos=[atompos[:,i] for i=1:length(atomtypes)]

    # Calculate the energy cutoff of the Fourier expansion.
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

    if ax == nothing fig,ax=subplots() end
    for i=1:sheets
        ax.plot(λ,evals[i,:],".",ms=2,c=colors[i])
    end
    ax.set_xticklabels(tick_labels)
    ax.set_xticks(λ[sympts_pos])
    ax.grid(axis="x",linestyle="dashed")
    ax.set_xlabel("High symmetry points")
    ax.set_ylabel("Total energy (Ry)")
    ax.set_title(name*" band structure plot")
    ax
end

@doc """
    plot_bandstructure(epm,ax,kpoint_dist,expansion_size,sheets)

Plot the band structure of an empirical pseudopotential.

# Arguments
- `epm::Union{epm₋model2D,epm₋model}`: an empirical pseudopotential.
- `ax::Union{PyObject,Nothing}=nothing`: an axes object from PyPlot.
- `kpoint_dist::Real`: the distance between k-points in the band plot.
- `expansion_size::Integer`: the desired number of terms in the Fourier
    expansion.
- `sheets::Int`: the sheets included in the electronic band structure plot.

# Returns
- `ax::PyCall.PyObject`: the band structure plot.
"""
function plot_bandstructure(epm::Union{epm₋model2D,epm₋model},
    ax::Union{PyObject,Nothing}=nothing; kpoint_dist::Real=0.01,
    expansion_size::Integer=1000, sheets::Integer=epm.sheets)
    plot_bandstructure(epm.name, epm.real_latvecs, epm.atom_types, epm.atom_pos,
        epm.rules, ax; expansion_size=expansion_size, sheets=sheets, 
        kpoint_dist=kpoint_dist, convention=epm.convention,
        coordinates=epm.coordinates)
end

@doc """
A dictionary whose keys are the labels of high symmetry points from the Python
package `seekpath`. The the values are the same labels but in a better-looking
format.
"""
labels_dict=Dict("GAMMA"=>"Γ","X"=>"X","U"=>"U","L"=>"L","W"=>"W","X"=>"X","K"=>"K",
                 "H"=>"H","N"=>"N","P"=>"P","Y"=>"Y","M"=>"M","A"=>"A","L_2"=>"L₂",
                 "V_2"=>"V₂","I_2"=>"I₂","I"=>"I","M_2"=>"M₂","Y"=>"Y",
                 "Z"=>"Z","Z_0"=>"Z₀","S"=>"S","S_0"=>"S₀","R"=>"R","G"=>"G",
                 "T"=>"T")

end # module
