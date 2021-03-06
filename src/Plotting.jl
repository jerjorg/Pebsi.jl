module Plotting

import PyPlot: subplots, figure, PyObject, figaspect
import QHull: chull

include("RectangularMethod.jl")
include("Polynomials.jl")

import .RectangularMethod: sample_unitcell
import .Polynomials: carttobary,barytocart,eval_poly,sample_simplex

import SymmetryReduceBZ.Plotting: plot_2Dconvexhull

@doc """
    meshplot(meshpts,ax,color)

Plot the points within a mesh in 2D or 3D.
"""
function meshplot(meshpts::AbstractArray{<:Real,2},
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
    contourplot(coeffs,simplex,ax)

Plot the level curves of a polynomial surface.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the polynomial.
- `simplex::AbstractArray{<:Real,2}`: the corners of the simplex as columns of
    the array.

# Returns
- `ax::PyObject`: the plot axes object.

"""
function contourplot(coeffs::AbstractArray{<:Real,1},
        simplex::AbstractArray{<:Real,2},ax::Union{PyObject,Nothing}=nothing,
        filled::Bool=false)
    dim = 2
    deg = 2
    ndivs = 1000
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
    shull = chull(Array(simplex'))

    if ax == nothing; (fig,ax)=subplots() end
    ax=plot_2Dconvexhull(shull,ax,"none")
    if filled 
        ax.contourf(X,Y,Z,[-1e10,0])
    else
        ax.contour(X,Y,Z,[0])            
    end
    ax
end

"""
    bezplot(bezpts)

Plot a quadratic Bezier curve and its Bezier points.
"""
function bezplot(bezpts::AbstractArray{<:Real,2},
        ax::Union{PyObject,Nothing}=nothing)
    
    dim = 1
    deg = 2
    simplex = [bezpts[1,1] bezpts[1,end]]
    s = 0.1*abs(simplex[2]-simplex[1]) 
    pts = Array(collect(simplex[1]-s:0.01:simplex[2]+s)')
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

end # module
