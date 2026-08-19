module AreaVolume

using SymmetryReduceBZ.Utilities
using SymmetryReduceBZ.Utilities: unique_points, shoelace, remove_duplicates, get_uniquefacets
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ...Polynomials: eval_poly,getpoly_coeffs,getbez_pts_wts,eval_bezcurve,
    conicsection, evalpoly1D, get_1Dquad_coeffs, solve_quadratic, bernstein_basis
using ...EPMs: eval_epm, EPM, EPM2D
using ...Mesh: get_neighbors, notbox_simplices, get_cvpts, ibz_init_mesh, 
    get_extmesh, choose_neighbors, choose_neighbors3D, trimesh, ntripts, ntetpts,
    get_sym_unique!, simplex_cornerpts, ibz_initmesh, ibz_borders,
    bz_translations
using ...Geometry: order_vertices!, simplex_size, insimplex, barytocart,
    carttobary, sample_simplex, lineseg_pt_dist, mapto_xyplane, ptface_mindist
using ...Defaults

using QHull: chull, Chull
using LinearAlgebra: cross, norm, dot, I, diagm, pinv, det
using Statistics: mean
using Base.Iterators: flatten
using PyCall: PyObject, pyimport
using Distributed: pmap
using FastGaussQuadrature: gausslegendre

using ..BezierSurfaces

export analytic_area, analytic_volume, two_intersects_area_volume, quad_area_volume

@doc """
    analytic_area(w::Real)

Calculate the area within a triangle and a canonical, rational Bezier curve.

The canonical triangle has corners at [-1,0], [1,0], and [0,1]. The weights of the
rational Bezier curve at corners [-1,0] and [1,0] are 1, so the only free parameter
is the weight of the middle Bezier point at [0,1]. See the notebook 
`analytic-expressions-derivation.nb` for a derivation of the analytic expression
and the Taylor expansion of the approximation.

# Arguments
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic Bezier
    curve.

# Returns
- `::Real`: the area within the triangle and Bezier curve.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: analytic_area
w = 1.0
analytic_area(w)
# output
0.6666666666666666
```
"""
function analytic_area(w::Real)::Real
    
    # Use the Taylor expansion of the analytic expression if the weight is close to 1.
    if isapprox(w,1,atol=def_taylor_exp_tol)
        2/3+4/15*(-1+w)-6/35*(-1+w)^2+32/315*(-1+w)^3-40/693*(-1+w)^4+(32*(-1+w)^5)/1001-
        (112*(-1+w)^6)/6435+ (1024*(-1+w)^7)/109395-(1152*(-1+w)^8)/230945+
        (2560*(-1+w)^9)/969969-(2816*(-1+w)^10)/2028117
    else
        a = sqrt(Complex(-1-w))
        b = sqrt(Complex(-1+w))
        abs(real((w*(w+(2*atan(b/a)/(a*b)))/(-1+w^2))))
    end
end

@doc """
    analytic_volume(coeffs,w)

Calculate the volume within a canonical triangle and Bezier curve of a quadratic surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic surface.
- `w::Real`: the weight of the middle Bezier point of a rational, quadratic Bezier curve.

# Returns
- `::Real`: the volume of the quadratic surface within the region bounded by the 
    triangle and the rational Bezier curve.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: analytic_volume
coeffs = [0.2,0.2,0.3,-0.3,0.4,-0.4]
w = 0.3
analytic_volume(coeffs,w)
# output
-0.029814783582691722
```
"""
function analytic_volume(coeffs::AbstractVector{<:Real},w::Real)::Real
    
    (c₅,c₃,c₀,c₄,c₁,c₂) = coeffs
    d = c₀+c₁+c₂+c₃+c₄+c₅
    # Use the Taylor expansion of the analytic solution if the weight is close to 1.
    if isapprox(w,1,atol=def_taylor_exp_tol)
        d/6*((6/7+(2*(-11*c₀-5*(c₁+c₃)+c₄))/(35*d))+4/105*(5+(3*c₀+5*(c₁+c₃)-c₄)/d)*(w-1)+(-(2/11)+(2*(81*c₀+
        5*(-5*(c₁+c₃)+c₄)))/(1155*d))*(w-1)^2+(32*(70+(-89*c₀+5*(-5*(c₁+c₃)+c₄))/d)*(w-1)^3)/15015+
        (8*(17*c₀-7*(c₁+6*c₂+c₃+7*c₄+6*c₅))*(w-1)^4)/(3003*d)+(64*(315+(-432*c₀+77*(-5*(c₁+c₃)+c₄))/
        d)*(w-1)^5)/255255+(224*(43*c₀+3*(30*c₁-55*c₂+30*c₃-72*c₄-55*c₅))*(w-1)^6)/(692835*d)+
        (1024*(165-(4*(46*c₀+75*(c₁+c₃)-15*c₄))/d)*(w-1)^7)/4849845-(384*(93*c₀-55*(41*c₁-39*c₂+41*c₃-55*c₄-39*c₅))*(w-1)^8)/
        (37182145*d)+(512*(1001+(-797*c₀+451*(-5*(c₁+c₃)+c₄))/d)*(w-1)^9)/37182145-
        (2816*(164*c₀+13*(-50*c₁+35*c₂-50*c₃+52*c₄+35*c₅))*(w-1)^10)/(152108775*d))
    else
        a = sqrt(Complex(-1-w))
        b = sqrt(Complex(-1+w))
        sign(w)real((w*(a*b*w*(-32*c₁+33*c₂-32*c₃+46*c₄+33*c₅-2*(-26*c₀+18*c₁+13*c₂+18*c₃+12*c₄+13*c₅)*w^2+
            8*d*w^4)+6*(5*c₂+6*c₄+5*c₅+4*(c₀-5*(c₁+c₃)+c₄)*w^2+16*c₀*w^4)*atan(b/a)))/(6*8*a*b*(-1+w^2)^3))
    end
end

@doc """
    simplex_extrema(coeffs;atol)

Return the exact minimum and maximum of a quadratic over its simplex.

A quadratic attains its extrema over a simplex either at a vertex, at a
stationary point of its restriction to an edge, or at its interior stationary
point, and all three are closed form. Sampling the interior instead - the
centroid, say - answers a different question, and gets it wrong exactly where the
level set passes through the point sampled.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the Bezier coefficients of the quadratic.
- `atol::Real`: a tolerance for placing a stationary point inside the simplex.

# Returns
- `Tuple{Real,Real}`: the minimum and maximum over the simplex.
"""
function simplex_extrema(coeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::Tuple{Real,Real}
    lo = minimum(coeffs[corner_indices]); hi = maximum(coeffs[corner_indices])
    for idx = ([1,2,3],[3,5,6],[6,4,1])
        c0,c1,c2 = coeffs[idx]
        A = c0 - 2*c1 + c2
        A == 0 && continue
        t = (c0 - c1)/A
        0 < t < 1 || continue
        v = (1-t)^2*c0 + 2*t*(1-t)*c1 + t^2*c2
        lo = min(lo,v); hi = max(hi,v)
    end
    stat = saddlepoint(coeffs,atol=atol)
    if all(isfinite,stat) && insimplex(stat)
        v = eval_poly(stat,coeffs,2,2)
        lo = min(lo,v); hi = max(hi,v)
    end
    lo,hi
end

@doc """
    two_intersects_area_volume(bezpts,quantity;atol)

Calculate the area or volume within a quadratic curve and triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of a quadratic surface.
- `quantity::String`: the quantity to compute ("area" or "volume").
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `areaₒᵣvolume::Real`: the area within the curve and triangle or the volume below
    the surface within the curve and triangle. The area is on the side of the curve
    where the surface is less than zero.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: two_intersects_area_volume
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.89 -0.08 -1.28 1.12 -0.081 -0.88]
two_intersects_area_volume(bezpts,"volume")
# output
-0.3533719907367465
```
"""
function two_intersects_area_volume(bezpts::AbstractMatrix{<:Real},
    quantity::String; atol::Real=def_atol,
    coeff_ref::Union{Nothing,Real}=nothing)::Real
     
    # Calculate the bezier curve and weights, make sure the curve passes through
    # the triangle
    triangle = bezpts[1:end-1,corner_indices]
    coeffs = bezpts[end,:]
    # What counts as zero for a coefficient is set by the coefficients present.
    # These carry the units of the interpolated quantity - eigenvalues measured
    # from the Fermi level - so a fixed threshold declares a patch entirely below
    # zero whenever its values happen to be small, which is exactly the patches
    # the Fermi surface passes through.
    cscale = maximum(abs, coeffs)
    # The reference is inherited when there is one, so a sub-patch made of
    # roundoff is judged against the scale the calculation started from.
    cref = coeff_ref === nothing ? cscale : coeff_ref
    # Same two floors as solve_quadratic: relative to the coefficients present,
    # but never below what this precision can resolve against the reference.
    ctol = max(cscale == 0 ? atol : def_coeff_rtol*cscale,
        def_coeff_noise_eps*eps(float(eltype(coeffs)))*abs(cref))
    # Whether the surface is one-signed over the whole simplex is a question about
    # its range, and the range is exact. Asking it first settles the cases the
    # intersection machinery cannot: a level set tangent to the simplex touches it
    # without enclosing anything, so the curve is found, two touch points are
    # collected, and a chord is drawn across a region that does not exist.
    #
    # It has to be the range and not a sample. The centroid of a surface tangent
    # along a line through it reads exactly zero, which says the simplex lies
    # below the level set when nothing does. It also has to be the range and not
    # the coefficients: those bound the surface without attaining it, so a
    # non-negative surface can still have a negative coefficient - x^2 over a
    # triangle has coefficients [1,-1,1,0,0,0] - and testing them alone reports a
    # region that is not there.
    lo,hi = simplex_extrema(coeffs,atol=atol)
    if lo >= -ctol
        return 0
    elseif hi <= ctol
        return quantity == "area" ? simplex_size(triangle) :
            quantity == "volume" ? mean(coeffs)*simplex_size(triangle) :
            throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
    end
    intersects = simplex_intersects(bezpts,atol=atol,coeff_ref=cref)
    # No intersections
    if intersects == [[],[],[]]
        # Case where the sheet is completely above or below 0.    
        if all(coeffs .< 0) && !any(isapprox.(coeffs,0,atol=ctol))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
            return areaₒᵣvolume
        end
        if all(coeffs .> 0) && !any(isapprox.(coeffs,0,atol=ctol))
            areaₒᵣvolume = 0
            return areaₒᵣvolume
        end
    end

    bezptsᵣ = []
    if intersects != [[],[],[]]
        all_intersects = reduce(hcat,[i for i=intersects if i != []])
        # `split_bezsurf` could not reduce this patch to two intersections.
        #
        # Below def_degenerate_simplex_size the patch is integrated by its sign
        # instead of exactly. What that costs is bounded by the patch itself - its
        # whole contribution to the area is its size, and to the volume its size
        # times its largest coefficient - so on a patch of 1e-16 the approximation
        # is worth less than the rounding already in the sum. Refusing it costs the
        # entire calculation, which is not a good trade for that.
        #
        # It warns rather than doing it quietly. This branch was made a hard error
        # earlier on the grounds that nothing could reach it: true of every 2D
        # path, and of a full refinement of m31, but 3D could not run at all at the
        # time and reaches it readily. An approximation that no one is told about
        # is the part worth avoiding, not the approximation.
        if size(all_intersects,2) != 2
            tsize = simplex_size(triangle)
            if tsize < def_degenerate_simplex_size
                @warn "Integrating a patch of size $(tsize) by its sign: it meets "*
                    "the triangle at $(size(all_intersects,2)) points and will not "*
                    "subdivide. The error this introduces is bounded by the patch's "*
                    "own contribution." maxlog=3
                if quantity == "area"
                    return mean(coeffs) < 0 ? tsize : 0
                elseif quantity == "volume"
                    return mean(coeffs) < 0 ? mean(coeffs)*tsize : 0
                else
                    throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
                end
            end
            error("Cannot integrate a patch of size $(tsize) that intersects the "*
                "triangle at $(size(all_intersects,2)) points and will not subdivide. "*
                "The patch is not small, so this indicates a genuine problem with "*
                "the geometry or the coefficients rather than a limit of precision.")
        end
        p₀ = all_intersects[:,1]
        p₂ = all_intersects[:,2]
        (bezptsᵣ,bezwtsᵣ) = getbez_pts_wts(bezpts,p₀,p₂,atol=atol)
        ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
        # Make sure the weight of the middle Bezier point has the correct sign.
        if !insimplex(carttobary(ptᵣ,triangle))
            bezwtsᵣ[2] *= -1
            ptᵣ = eval_bezcurve(0.5,bezptsᵣ,bezwtsᵣ)
            if !insimplex(carttobary(ptᵣ,triangle))
                intersects = [[],[],[]]
            end
        else
            # Remove intersections if the mipoint of the Bezier curve is on an edge.
            # Whether the curve's midpoint lies on an edge is a distance in
            # coordinate space, so the tolerance carries the triangle's units.
            # Held fixed it stops asking whether the point is on the edge and
            # starts asking whether the triangle is small: on a patch of extent
            # 1e-7, atol of 1e-9 is a hundredth of the whole patch, and points
            # well inside it are discarded along with their intersections.
            tex = maximum(maximum(triangle,dims=2) - minimum(triangle,dims=2))
            etol = tex > 0 ? atol*tex : atol
            on_edge = any(isapprox.([lineseg_pt_dist(ptᵣ,triangle[:,i],atol=etol) 
                for i=[[1,2],[2,3],[3,1]]],0,atol=etol))
            if on_edge intersects = [[],[],[]] end
        end
    end

    # If the tangent lines are close to parallel, the middle Bezier point of the
    # curve will be very far away, which introduces numerical errors. We handle
    # this by splitting the surface up and recalculating.
    cstype = conicsection(coeffs) # using the default tolerance of 1e-12    
    linear = any(cstype .== ["line","rectangular hyperbola","parallel lines"])
    split = false
    if bezptsᵣ != []
        if maximum(abs.(bezptsᵣ)) > def_rational_bezpt_dist 
            split = true
        end
    end

    # Split the triangle if the saddle point is within the triangle but not on a
    # corner and the conic section is linear or degenerate.
    saddle = saddlepoint(coeffs,atol=atol)
    if insimplex(saddle) && !any([isapprox(saddle,x,atol=atol) for x=[[1,0,0],[0,1,0],[0,0,1]]])
        split = true
    end

    if split
        bezptsᵤ = [split_bezsurf(b,atol=atol,coeff_ref=cref) for b=split_bezsurf₁(bezpts,coeff_ref=cref)] |> flatten |> collect
        return sum([two_intersects_area_volume(b,quantity,atol=atol,coeff_ref=cref) for b=bezptsᵤ])
    end

    # No intersections, no island, and the coefficients are less or greater than 0.
    if intersects == [[],[],[]]
        v = eval_poly([1/3,1/3,1/3],coeffs,2,2)
        if v < 0 || isapprox(v,0,atol=ctol)
            below = true
        else
            below = false
        end

        if below
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
        else
            areaₒᵣvolume = 0
        end
        return areaₒᵣvolume
    end
    edgesᵢ = [i for i=1:3 if intersects[i] != []]
    if length(edgesᵢ) == 1
        # When intersections are on two different edges, we need to include the
        # area or volume from a subtriangle in addition to the canonical rational
        # Bezier triangle and the whole triangle. It has no effect when the intersections
        # are on the same edge.
        corner = [1,2,3][edgesᵢ[1]]

        # If two intersections on the same edge, use a point that is the average of 
        # the intersections and the midpoint of the Bezier curve.
        avept = mean(all_intersects,dims=2) |> vec
        avept = mean([avept ptᵣ],dims=2) |> vec
    elseif length(edgesᵢ) ==2
        corner = [3,1,2][setdiff([1,2,3],edgesᵢ)[1]]
        # If intersections on different edges, use a point that is the average of
        # the intersections and the corner.
        avept = mean([ptᵣ triangle[:,corner]],dims=2) |> vec
    else
        error("The curve may only intersect at most two edges.")
    end
    avept = carttobary(avept,triangle)
    if (eval_poly(avept,coeffs,2,2) < 0 || 
        isapprox(eval_poly(avept,coeffs,2,2), 0, atol=ctol))
        below₀ = true
    else
        below₀ = false
    end 

    simplex_bpts = sample_simplex(2,2)
    triangleₑ = order_vertices!([all_intersects triangle[:,corner]])
    if quantity == "area"
        # curve area or volume
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_area(bezwtsᵣ[2])
    elseif quantity == "volume"
        coeffsᵣ = sub_coeffs(bezpts,bezptsᵣ)
        areaₒᵣvolumeᵣ = simplex_size(bezptsᵣ)*analytic_volume(coeffsᵣ,bezwtsᵣ[2])
    else
        throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
    end

    # Get the sign of the area correct (accounts for the curvature of the curve).
    inside = false
    # Get exception when corners of triangleₑ all lie on a straight line.
    try
        inside = insimplex(carttobary(ptᵣ,triangleₑ))
    catch SingularException
        nothing
    end

    if length(edgesᵢ) == 2 && inside
        areaₒᵣvolumeᵣ *= -1
    end
    
    if quantity == "area"
        areaₒᵣvolume =  areaₒᵣvolumeᵣ + simplex_size(triangleₑ)
        if !below₀
            areaₒᵣvolume = simplex_size(triangle) - areaₒᵣvolume
        end
    else # quantity == "volume"
        coeffsₑ = sub_coeffs(bezpts,triangleₑ)
        areaₒᵣvolume = mean(coeffsₑ)*simplex_size(triangleₑ) + areaₒᵣvolumeᵣ
        if !below₀
            areaₒᵣvolume = simplex_size(triangle)*mean(coeffs) - areaₒᵣvolume
        end
    end

    areaₒᵣvolume
end

@doc """
    quad_area_volume(bezpts,quantity;num_slices,atol)

Calculate the area of the shadow or the volume beneath a quadratic.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface in
    columns of a matrix.
- `quantity::String`: the quantity to calculate ("area" or "volume").
- `num_slices::Integer`: a dummy variable to simplify 2D and 3D integration.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `::Real`: the area of the shadow of a quadratic polynomial within a triangle
    and below the plane `z=0` or the volume of the quadratic polynomial under the 
    same constraints.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: quad_area_volume
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 2/3 -4/3 2/3 -2/3 -2/3 0]
quad_area_volume(bezpts,"area") ≈ 0.8696051011068969
# output
true
```
"""
function quad_area_volume(bezpts::AbstractMatrix{<:Real},
        quantity::String; num_slices::Integer=def_num_slices, atol::Real=def_atol)::Real
     
    # Modifications to make when working in 3D.
    if size(bezpts,1) == 4
        bezpts = [mapto_xyplane(bezpts[1:3,:]); bezpts[end,:]']
    end
    # The top of the chain sets the reference: every sub-patch below is judged
    # against the coefficients of the patch actually handed in.
    cref = maximum(abs, bezpts[end,:])
    sum([two_intersects_area_volume(b,quantity,atol=atol,coeff_ref=cref) for 
        b=split_bezsurf(bezpts,atol=atol,coeff_ref=cref)])
end

end # module AreaVolume
