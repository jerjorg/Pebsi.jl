module Plotting

using PyPlot: subplots, figure, PyObject, figaspect, plt, pyimport

using QHull: chull

using ..QuadraticIntegration: bandstructure
using ..EPMs: epm₋model2D
using ..RectangularMethod: sample_unitcell
using ..Polynomials: eval_poly,sample_simplex,eval_bezcurve
using ..Geometry: carttobary,barytocart,simplex_size

using SymmetryReduceBZ.Plotting: plot_2Dconvexhull
using SymmetryReduceBZ.Utilities: sortpts2D

using Statistics: mean

@doc """
    meshplot(meshpts,ax,color)

Plot the points within a mesh in 2D or 3D.
"""
function meshplot(meshpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; color::String="blue", alpha=0.5,
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
        ax.auto_scale_xyz(plotrange[1],plotrange[2],plotrange[3])o

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
    padded::Bool=true,ndivs::Integer=100,colors=["black","white"],
    alpha::Real=0.5,curvewidths=1,zorder::Int=1)::PyObject
    dim = 2
    deg = 2
    # Plot triangle
    coeffs = bezpts[3,:]
    simplex = bezpts[1:2,[1,3,6]]
    if ax == nothing; (fig,ax)=subplots() end
    if padded
        N = [ndivs 0; 0 ndivs]
        basis = [simplex[:,2] - simplex[:,1] simplex[:,3] - simplex[:,1]]
        grid_offset = [0.5,0.5]
        plotpts = 2 .* sample_unitcell(basis,N,grid_offset)
        plotpts = mapslices(x->x-basis*[1/2,1/2]+simplex[:,1],plotpts,dims=1)
        bplotpts = carttobary(plotpts,simplex)
        plotvals= eval_poly(bplotpts,coeffs,dim,deg)
        X=reshape(plotpts[1,:],(ndivs,ndivs))
        Y=reshape(plotpts[2,:],(ndivs,ndivs))
        Z=reshape(plotvals,(ndivs,ndivs))

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
    sort=false, facecolor::String="None", alpha_tri::Real=1, linewidth::Real=0.5,
    edgecolor::String="black", ndiv::Integer=100, filled::Bool=false,
    ndivs::Integer=100, colors=["black","white"], alpha_curve::Real=0.5,
    curvewidths::Real=1)::PyObject 
     
    patch=pyimport("matplotlib.patches")
    collections=pyimport("matplotlib.collections")

    bpts = sample_simplex(2,2);
    tripts = [Array(ebs.mesh.points[i,:]') for i=ebs.simplicesᵢ];

    if ax == nothing fig,ax=subplots() end

    # Locate triangles and sheets that are partially occupied
    pocc = [findall(x->x==1,b) for b=ebs.partially_occupied]
    indices = reduce(vcat,
        filter(x->!(x==nothing),
            [pocc[i] == [] ? nothing : [[i,j] for j=pocc[i]] for i=1:length(pocc)]));

    sizes = [simplex_size(t) for t=tripts]
    ρ = ndiv/maximum(sizes)
    ndivs = round.(Int,ρ*sizes)

    for i=indices
        bezpts = [barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[i[1]],:]'); 
            mean(ebs.mesh_intcoeffs[i[1]][i[2]],dims=1) .- ebs.fermilevel]
        ax = contourplot(bezpts,ax,padded=false,ndivs=ndivs[i[1]],
            alpha=alpha_curve,curvewidths=curvewidths,colors=colors,
            filled=filled,zorder=2)
    end

    # Plot the triangles.
    ax = polygonplot(tripts,ax,facecolor=facecolor,sort=sort,alpha=alpha_tri,
        linewidth=linewidth,edgecolor=edgecolor,zorder=1) 

    # Set plot range.
    xrange = [minimum(ebs.mesh.points[5:end,1]), maximum(ebs.mesh.points[5:end,1])]
    xrange = [xrange[1] - 0.05*diff(xrange)[1], xrange[2] + 0.05*diff(xrange)[1]]
    yrange = [minimum(ebs.mesh.points[5:end,2]), maximum(ebs.mesh.points[5:end,2])]
    yrange = [yrange[1] - 0.05*diff(yrange)[1], yrange[2] + 0.05*diff(yrange)[1]]
    ax.set_xlim(xrange)
    ax.set_ylim(yrange)        
    ax
end

# function contourplot(bezpts::AbstractMatrix{<:Real},
#     ax::Union{PyObject,Nothing}=nothing; filled::Bool=false,
#     padded::Bool=true,ndivs::Integer=100,colors=["black","white"],
#     alpha::Real=0.5,curvewidths=1)::PyObject


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
    ax::Union{PyObject,Nothing}=nothing; sort=false, facecolor::String="None",
    alpha::Real=1.0, linewidth::Real=0.5, edgecolor::String="black",
    zorder::Int=0)::PyObject

    if sort
        perm = sortpts2D(pts)
        pts = pts[:,perm]
    end
    if ax == nothing fig,ax = subplots() end
    # ax.add_patch(plt.Polygon(Array(pts'),edgecolor=edgecolor,facecolor=facecolor,
    #     alpha=alpha,linewidth=linewidth))
    ax.fill(pts[1,:],pts[2,:],edgecolor=edgecolor,facecolor=facecolor,
        alpha=alpha,linewidth=linewidth,zorder=zorder)
    ax
end

function polygonplot(polygons::Vector{Matrix{T}} where T<:Real,
    ax::Union{PyObject,Nothing}=nothing;sort=false, facecolor::String="None",
    alpha::Real=1.0, linewidth::Real=3, edgecolor::String="black", zorder::Int=0)
    if ax == nothing fig,ax = subplots() end

    for poly in polygons
        ax = polygonplot(poly,ax;sort=sort,facecolor=facecolor,alpha=alpha,
            linewidth=linewidth,edgecolor=edgecolor,zorder=zorder)
    end
    ax
end
end # module
