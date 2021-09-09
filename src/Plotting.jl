module Plotting

using PyPlot: subplots, figure, PyObject, figaspect, plt
using QHull: chull

using ..QuadraticIntegration: bandstructure
using ..EPMs: epm₋model2D
using ..RectangularMethod: sample_unitcell
using ..Polynomials: eval_poly,sample_simplex,eval_bezcurve
using ..Geometry: carttobary,barytocart

using SymmetryReduceBZ.Plotting: plot_2Dconvexhull

using Statistics: mean

@doc """
    meshplot(meshpts,ax,color)

Plot the points within a mesh in 2D or 3D.
"""
function meshplot(meshpts::AbstractMatrix{<:Real},
    ax::Union{PyObject,Nothing}=nothing; color::String="blue", alpha=0.5)

    dim = size(meshpts,1)
    if dim == 2
        if ax == nothing
            (fig,ax) = subplots()
        end
        ϵ=0.1*max(meshpts...)

        (xs,ys)=[meshpts[i,:] for i=1:dim]
        ax.scatter(xs,ys,color=color,alpha=alpha)
        ax.set_aspect(1)

    elseif dim == 3
        if ax == nothing
            fig = figure()
            ax = fig.add_subplot(111, projection="3d")
        end
        (xs,ys,zs)=[meshpts[i,:] for i=1:dim]
        ax.scatter3D(xs,ys,zs,color=color,alpha=alpha)

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
    meshplot(epm,ebs,ax)

Plot the triangles in a mesh.
"""
function meshplot(epm::epm₋model2D,ebs::bandstructure,ax::Union{PyObject,Nothing}=nothing)

    if ax == nothing
        (fig,ax) = subplots()
    end
    
    for s = ebs.simplicesᵢ
        ax.add_patch(plt.Polygon(ebs.mesh.points[s,:],edgecolor="gray",facecolor="none"))
    end
    ax = meshplot(ebs.mesh.points[5:end,:]',ax,color="black")
    ax = plot_2Dconvexhull(epm.ibz,ax,facecolor="none",edgecolor="blue")
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
    padded::Bool=true)::PyObject

    dim = 2
    deg = 2
    ndivs = 100
    # Plot triangle
    coeffs = bezpts[3,:]
    simplex = bezpts[1:2,[1,3,6]]
    if ax == nothing; (fig,ax)=subplots() end
    shull = chull(Array(simplex'))
    ax=plot_2Dconvexhull(shull,ax,facecolor="None")
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
            ax.contourf(X,Y,Z,[-1e10,0],colors=["black","white"],alpha=0.5)
        else
            ax.contour(X,Y,Z,[0],colors=["blue"])         
        end
    else
        bpts = sample_simplex(2,100)
        pts = barytocart(bpts,simplex)
        vals = eval_poly(bpts,coeffs,dim,deg)
        if filled
            ax.tricontourf(pts[1,:],pts[2,:],vals,[-1e19,0],colors=["black","white"],
                alpha=0.5)
        else
            ax.tricontour(pts[1,:],pts[2,:],vals,[0],[0],colors=["red"])
        end
    end
    ax
end

function contourplot(epm::epm₋model2D,ebs::bandstructure,ax::Union{PyObject,Nothing}=nothing)
    bpts = sample_simplex(2,2,)

    if ax == nothing
        (fig,ax) = subplots()
    end
    
    for i=1:length(ebs.simplicesᵢ)
        for j=1:epm.sheets
            bezpts = [barytocart(bpts,ebs.mesh.points[ebs.simplicesᵢ[i],:]'); 
                mean(ebs.mesh_intcoeffs[i][j],dims=1) .- ebs.fermilevel]
            ax = contourplot(bezpts,ax,padded=false)
        end
    end
    ax=meshplot(epm,ebs,ax)
    ax = meshplot(ebs.mesh.points[5:end,:]',ax)

    ax
end

"""
    bezplot(bezpts)

Plot a quadratic Bezier curve and its Bezier points (1D).
"""
function bezplot(bezpts::AbstractMatrix{<:Real},
        ax::Union{PyObject,Nothing}=nothing)
    
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
    ax::Union{PyObject,Nothing}=nothing)::PyObject
    if ax == nothing; (fig,ax)=subplots(); end
    
    data = eval_bezcurve(collect(0:1/1000:1),bezptsᵣ,bezwtsᵣ)
    ax = meshplot(bezptsᵣ,ax)
    ax.plot(data[1,:],data[2,:],color="blue")
    ax
end

end # module
