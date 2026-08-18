@doc """
    quadval_vertex(bezcoeffs)

Calculate the value of a 1D quadratic curve at its vertex.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the quadratic polynomial coefficients.

# Returns
- `::Real`: the maximum or minimum value of the quadratic polynomial.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: quadval_vertex
coeffs = [-1, 2, -1]
quadval_vertex(coeffs)
# output
0.5
```
"""
function quadval_vertex(bezcoeffs::AbstractVector{<:Real})::Real
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
end

@doc """
The locations of the quadratic Bezier points at the corners of the triangle in
counterclockwise order for 2D quadratic Bezier points.
"""
corner_indices = [1,3,6]

@doc """
The locations of quadratic Bezier points along each edge of the triangle in
counterclockwise order for 2D quadratic Bezier points.
"""
edge_indices=[[1,2,3],[3,5,6],[6,4,1]]

@doc """
    simplex_intersects(bezpts,atol)

Find the locations where a level curve of a quadratic surface intersects a triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic Bezier surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `intersects::Array`: the intersections organized by edge in a vector. Each element
    of the vector is a matrix where the columns are the Cartesian coordinates of
    theintersections.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simplex_intersects
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -1.0 -2.0 1.0 0.0 2.0]
simplex_intersects(bezpts)
# output
3-element Vector{Array}:
 [-1.0; 0.0;;]
 [0.5; 0.5;;]
 Any[]
```
"""
function simplex_intersects(bezpts::AbstractMatrix{<:Real};
    atol::Real=def_atol)::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = bezcurve_intersects(edge_bezpts[end,:];atol=atol)
        if edge_ints != []
            intersects[i] = reduce(hcat,[edge_bezpts[1:end-1,1] .+ 
                i*(edge_bezpts[1:end-1,end] .- edge_bezpts[1:end-1,1]) for i=edge_ints])
        end
    end
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects == 1
        Array{Array,1}([[],[],[]])
    else
        intersects
    end
end

@doc """
    saddlepoint(coeffs;atol)

Calculate the saddle point of a quadratic Bezier surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `::AbstractVector{<:Real}`: the coordinates of the saddle point in barycentric
    coordinates.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: saddlepoint
coeffs = [0.36, -1.64, 0.36, -0.64, -0.64, 0.36]
saddlepoint(coeffs)
# output
3-element Vector{Float64}:
 0.5000000000000001
 0.5000000000000001
 4.163336342344338e-17
```
"""
function saddlepoint(coeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector{<:Real}
    # (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀) = coeffs
    (z₂₀₀, z₁₁₀, z₀₂₀, z₁₀₁, z₀₁₁, z₀₀₂) = coeffs
    denom = z₀₁₁^2+(z₁₀₁-z₁₁₀)^2+z₀₂₀*(2z₁₀₁-z₂₀₀)-2z₀₁₁*(z₁₀₁+z₁₁₀-z₂₀₀)-z₀₀₂*(z₀₂₀-2z₁₁₀+z₂₀₀)
    
    if isapprox(denom,0,atol=atol)
        return [Inf,Inf,Inf]
    end
    sₑ = z₀₁₁^2+z₀₂₀*z₁₀₁+z₀₀₂*(-z₀₂₀+z₁₁₀)-z₀₁₁*(z₁₀₁+z₁₁₀)
    tₑ = -z₁₀₁*(z₀₁₁-z₁₀₁+z₁₁₀)+z₀₀₂*(z₁₁₀-z₂₀₀)+z₀₁₁*z₂₀₀
    uₑ = -(z₀₁₁+z₁₀₁-z₁₁₀)*z₁₁₀+z₀₂₀*(z₁₀₁-z₂₀₀)+z₀₁₁*z₂₀₀
    [sₑ,tₑ,uₑ]/denom
end

@doc """
    split_bezsurf₁(bezpts,atol)

Split a Bezier surface once into sub-Bezier surfaces with the Delaunay method.

A triangular mesh is created using a Delaunay tesselation of points at the corners 
of the triangle, the midpoints of the edges, the intersections of the level curves
of the quadratic with the triangle, and the double point of the quadratic surface
if it lies within the triangle. Bezier coefficients for these simplices are calculated.
The goal is to split the quadratic surface into subsurfaces that have no more than
two intersections of the level curves with the edges of the triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces. The sub-surfaces
    reproduce the original Bezier surface.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_bezsurf₁
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 0.0 1.0 -1.0 0.0]
sbezpts = split_bezsurf₁(bezpts)
length(sbezpts)
# output
6
```
"""
function split_bezsurf₁(bezpts::AbstractMatrix{<:Real}; atol::Real=def_atol)::AbstractArray
    spatial = pyimport("scipy.spatial")
    dim = 2; deg = 2; 
    triangle = bezpts[1:end-1,corner_indices]
    if simplex_size(triangle) < def_min_simplex_size
        return [bezpts]
    end
     
    coeffs = bezpts[end,:]; pts = bezpts[1:end-1,:]
    simplex_bpts = sample_simplex(dim,deg)
    intersects = simplex_intersects(bezpts,atol=atol)
    spt = saddlepoint(coeffs)
    allpts = pts
    if insimplex(spt) # Using the default absolute tolerance 1e-12
        allpts = [pts barytocart(spt,triangle)]
    end
    if intersects != [[],[],[]]
        allintersects = reduce(hcat,[i for i=intersects if i!=[]])
        allpts = [allpts allintersects]
    end
    allpts = unique_points(allpts,atol=atol)
    # Had to add box points to prevent collinear triangles.
    xmax,ymax = maximum(bezpts[1:2,:],dims=2)
    xmin,ymin = minimum(bezpts[1:2,:],dims=2)
    xmax += def_mesh_scale*abs(xmax - xmin)
    xmin -= def_mesh_scale*abs(xmax - xmin)
    ymax += def_mesh_scale*abs(ymax - ymin)
    ymin -= def_mesh_scale*abs(ymax - ymin)
    boxpts = [xmin xmax xmax xmin; ymin ymin ymax ymax]
    allpts = [boxpts allpts]
    del = spatial.Delaunay(Matrix(allpts'))
    tri_ind = notbox_simplices(del)
    # For small triangles, all triangles may have a corner at a box corner.
    # In this case, return the original points. Return `bezpts` rather than
    # `pts`: callers expect the coefficients in the last row, and dropping it
    # leaves a matrix one row short that fails downstream in `simplex_size`.
    if length(tri_ind) == 0
        return [bezpts]
    end
    tri_ind = reduce(hcat,tri_ind)
    subtri = [order_vertices!(allpts[:,tri_ind[:,i]]) for i=1:size(tri_ind,2)]
    sub_pts = [barytocart(simplex_bpts,tri) for tri=subtri]
    sub_bpts = [carttobary(pts,triangle) for pts=sub_pts]
    sub_vals = [reduce(hcat, [eval_poly(sub_bpts[j][:,i],coeffs,dim,deg)
        for i=1:6]) for j=1:length(subtri)]
    subtri_coeffs = [getpoly_coeffs(v[:],simplex_bpts,dim,deg) for v=sub_vals]
    sub_bezpts = [[sub_pts[i]; subtri_coeffs[i]'] for i=1:length(subtri_coeffs)]
    sub_bezpts
end

@doc """
    split_bezsurf(bezpts;atol)

Split a Bezier surface into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic surface.
- `atol::Real=def_atol`: an absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces. The sub-surfaces
    reproduce the original surface.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_bezsurf
bezpts = [0. 0.5 1 0.5 1 1; 0. 0. 0. 0.5 0.5 1; 1.1 1.2 -1.3 1.4 1.5 1.6]
split_bezsurf(bezpts)
# output
1-element Vector{Matrix{Float64}}:
 [0.0 0.5 … 1.0 1.0; 0.0 0.0 … 0.5 1.0; 1.1 1.2 … 1.5 1.6]
```
*See `split_bezsurf₁` for a more detailed description.
"""
function split_bezsurf(bezpts::AbstractMatrix{<:Real};atol=def_atol)::AbstractArray
    
    intersects = simplex_intersects(bezpts,atol=atol)
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects <= 2
        return [bezpts]
    else
        sub_bezpts = split_bezsurf₁(bezpts)
        sub_intersects = [simplex_intersects(b,atol=atol) for b=sub_bezpts]
        num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
            size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
        while any(num_intersects .> 2)
            # `split_bezsurf₁` returns its input unchanged when the surface
            # cannot be subdivided any further (a degenerate or very small
            # triangle). Re-splitting such a surface makes no progress, so track
            # whether any split succeeded and stop when none did — otherwise
            # this loop never terminates.
            split_occurred = false
            for i = length(num_intersects):-1:1
                if num_intersects[i] <= 2 continue end
                subs = split_bezsurf₁(sub_bezpts[i])
                if length(subs) == 1 continue end
                split_occurred = true
                append!(sub_bezpts,subs)
                deleteat!(sub_bezpts,i)
                sub_intersects = [simplex_intersects(b,atol=atol) for b=sub_bezpts]
                num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 :
                    size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
            end
            if !split_occurred break end
        end
    end
    sub_bezpts
end

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
    sub_coeffs(bezpts,subtriangle)

Calculate the coefficients of a quadratic sub-surface of a quadratic triangle.

# Arguments
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points of the quadratic triangle.
- `subtriangle::AbstractMatrix{<:Real}`: the points at the corners of a subtriangle
    as columns of an array.

# Returns
- `::AbstractVector{<:Real}`: the Bezier coefficients that give a subsurface of 
    a quadratic surface. The subsurface has a domain of `subtriangle`.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: sub_coeffs
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; -0.25 -0.25 3.75 -0.25 1.75 1.75]
subtriangle = [-0.5 0.0 -0.6464466094067263; 0.0 1.0 0.35355339059327373]
round.(sub_coeffs(bezpts,subtriangle), digits=10)
# output
6-element Vector{Float64}:
  0.0
  0.25
  1.75
 -0.0732233047
  0.4571067812
 -0.0
```
"""
function sub_coeffs(bezpts::AbstractMatrix{<:Real},
    subtriangle::AbstractMatrix{<:Real})::AbstractVector{<:Real}

    ptsᵢ = carttobary(barytocart(sample_simplex(2,2),subtriangle),bezpts[1:2,corner_indices])
    valsᵢ = eval_poly(ptsᵢ,bezpts[end,:],2,2)
    getpoly_coeffs(valsᵢ,sample_simplex(2,2),2,2)
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
    quantity::String; atol::Real=def_atol)::Real
     
    # Calculate the bezier curve and weights, make sure the curve passes through
    # the triangle
    triangle = bezpts[1:end-1,corner_indices]
    coeffs = bezpts[end,:]
    intersects = simplex_intersects(bezpts,atol=atol)
    # No intersections
    if intersects == [[],[],[]]
        # Case where the sheet is completely above or below 0.    
        if all(coeffs .< 0) && !any(isapprox.(coeffs,0,atol=atol))
            if quantity == "area"
                areaₒᵣvolume = simplex_size(triangle)
            elseif quantity == "volume"
                areaₒᵣvolume = mean(coeffs)*simplex_size(triangle)
            else
                throw(ArgumentError("The quantity calculated is either \"area\" or \"volume\"."))
            end
            return areaₒᵣvolume
        end
        if all(coeffs .> 0) && !any(isapprox.(coeffs,0,atol=atol))
            areaₒᵣvolume = 0
            return areaₒᵣvolume
        end
    end

    bezptsᵣ = []
    if intersects != [[],[],[]]
        all_intersects = reduce(hcat,[i for i=intersects if i != []])
        # Known limitation, reachable via the `m31` EPM: `split_bezsurf` can hand
        # back a surface with three intersections when it cannot subdivide the
        # triangle any further. The trigger is a triangle of area ~3e-9, where
        # the box-padded Delaunay triangulation in `split_bezsurf1` produces only
        # triangles with a corner on the padding box, so `tri_ind` is empty.
        # `def_min_simplex_size` (1e-12) is far too small to catch this. Deciding
        # how such a patch should contribute is a question about the integration
        # method, not a typo, so it is left erroring rather than papered over.
        if size(all_intersects,2) != 2
            error("Can only calculate the area or volume when the curve intersects
                the triangle at two points or doesn't intersect the triangle.")
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
            on_edge = any(isapprox.([lineseg_pt_dist(ptᵣ,triangle[:,i],atol=atol) 
                for i=[[1,2],[2,3],[3,1]]],0,atol=atol))
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
        bezptsᵤ = [split_bezsurf(b,atol=atol) for b=split_bezsurf₁(bezpts)] |> flatten |> collect
        return sum([two_intersects_area_volume(b,quantity,atol=atol) for b=bezptsᵤ])
    end

    # No intersections, no island, and the coefficients are less or greater than 0.
    if intersects == [[],[],[]]
        v = eval_poly([1/3,1/3,1/3],coeffs,2,2)
        if v < 0 || isapprox(v,0,atol=atol)
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
        isapprox(eval_poly(avept,coeffs,2,2), 0, atol=atol))
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
    sum([two_intersects_area_volume(b,quantity,atol=atol) for 
        b=split_bezsurf(bezpts,atol=atol)])
end

@doc """
    bezcurve_intersects(bezcoeffs;rtol,atol)

Determine where a quadratic curve is equal to zero.

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `solutions::AbstractVector`: the locations between [0,1) where the quadratic
    equals 0.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: bezcurve_intersects
coeffs = [0,1,-1]
bezcurve_intersects(coeffs)
# output
2-element Vector{Float64}:
 0.0
 0.6666666666666666
```
"""
function bezcurve_intersects(bezcoeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector
    a,b,c = bezcoeffs
    solutions = solve_quadratic(a - 2*b + c, 2*(b-a), a)
    solutions = filter(t -> (t > 0 || isapprox(t,0,atol=atol)) 
        && (t < 1 && !isapprox(t,1,atol=atol)), solutions)
    return solutions
end

@doc """
    getdomain(bezcoeffs;atol)

Calculate the interval(s) of a quadratic where it is less than 0 between (0,1).

# Arguments
- `bezcoeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `atol::Real=def_atol`: an absolute tolerance for floating point comparisons.

# Returns
- `reg::AbstractVector`: the region(s) where the quadratic is less than 0.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: getdomain
coeffs = [0,1,-1]
getdomain(coeffs)
# output
2-element Vector{Any}:
 0.6666666666666666
 1.0
```
"""
function getdomain(bezcoeffs::AbstractVector{<:Real};
    atol::Real=def_atol)::AbstractVector
    vals = [evalpoly1D(t,bezcoeffs) for t=[0,1/2,1]]
    if all(vals .< 0) && !any(isapprox.(vals,0,atol=atol))
        return [0,1]
    elseif all(vals .> 0) && !any(isapprox.(vals,0,atol=atol))
        return []
    end
    intersects = bezcurve_intersects(bezcoeffs)
    shaded = [[0,1]; intersects] |> remove_duplicates |> sort
    
    reg = []
    for i = 1:length(shaded) - 1
        test_pts = collect(range(shaded[i],shaded[i+1],step=(shaded[i+1]-shaded[i])/10))
        test_vals = [evalpoly1D(t,bezcoeffs) for t=test_pts]
        if all(x -> x < 0 || isapprox(x,0,atol=atol),test_vals)
            reg = [reg; shaded[[i,i+1]]]
        end
    end
    if reg != []
        reg = reg |> remove_duplicates |> sort
    end
    
    if length(reg) == 3
        if isapprox(sum(diff(reg)),1,atol=atol)
            reg = [0,1]
        end
    end
    reg
end

@doc """
    analytic_area1D(coeffs,limits)

Calculate the area of a quadratic where it is less than zero between [0,1].

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `limits::AbstractVector`: the interval(s) where the quadratic is less 
    than zero.

# Returns
- `area::Real`: the area under the quadratic

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: analytic_area1D
coeffs = [0.,1.,-1.]
limits = [0.,1.]
analytic_area1D(coeffs,limits)
# output
0.0
```
"""
function analytic_area1D(coeffs::AbstractVector{<:Real}, limits::AbstractVector)::Real
    if length(limits) == 0
        area = 0.
    elseif length(limits) == 2
        a,b,c = coeffs
        t0,t1 = limits
        area = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
          3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
    elseif length(limits) == 4
        a,b,c = coeffs
        t0,t1 = limits[1:2]
        area1 = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
                  3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
        t0,t1 = limits[3:4]
        area2 = 1/3*(t0*(-3*a + 3*(a - b)*t0 - (a - 2*b + c)*t0^2) + 3*a*t1 + 
                  3*(-a + b)*t1^2 + (a - 2*b + c)*t1^3)
        area = area1 + area2
    else
        error("More limits than expected in the 1D quadratic area calculation.")
    end
    area
end

@doc """
    simpson(y,int_len)

Integrate the area below a curve with Simpson's method.

# Arguments
- `y::AbstractVector{<:Real}`: a list of values of the curve being integrated.
- `int_len::Real`: the length of the interval over which the curve is integrated.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simpson
f(x)=x^3+x^2+1
v=map(x->f(x),range(-1,3,step=0.1))
simpson(v,4)
# output
33.333333333333336
``` 
"""
function simpson(y::AbstractVector{<:Real},int_len::Real)::Real
    n = length(y)-1
    n % 2 == 0 || error("The number of intervals must be odd for Simpson's method.")
    int_len/(3n) * sum(y[1:2:n] + 4*y[2:2:n] + y[3:2:n+1])
end

@doc """
    linept_dist(line,pt)

Calculate the shortest distance between a point and a line embedded in 2D.

# Arguments
- `line::Matrix{<:Real}`: the endpoints of a line segment as columns of an matrix.
- `pt::Vector{<:Real}`: the coordinates of a point in a vector

# Example
```jldoctest
using Pebsi.QuadraticIntegration: linept_dist
line = [0 1; 0 0]
pt = [0,2]
linept_dist(line,pt)
# output
2.0
```
"""
function linept_dist(line,pt)::Real
    unit_vec = [0 -1; 1 0]*(line[:,2] - line[:,1])/norm(line[:,2] - line[:,1])
    abs(dot(unit_vec,pt-line[:,1]))
end

face_ind = [[2,3,4],[1,3,4],[1,4,2],[1,2,3]]
corner_ind = [1,2,3,4]

# Labeled by corner opposite the face
# The order of the sample points of a slice of the tetrahedron
slice_order1 = [4,2,3,1]
slice_order2 = [1,4,2,3]
slice_order3 = [1,3,4,2]
slice_order4 = [1,2,3,4]

# The order of the coefficients of the 3D quadratic polynomial when the slices
# area towards different corners
coeff_order1 = [3, 8, 10, 5, 9, 6, 2, 7, 4, 1]
coeff_order2 = [1, 4, 6, 7, 9, 10, 2, 5, 8, 3]
coeff_order3 = [1, 7, 10, 2, 8, 3, 4, 9, 5, 6]
coeff_order4 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# The order of the vertices of the tetrahedron when the slices are taken towards
# different corners of the tetrahedron
vert_order1 = [4,1,3,2]
vert_order2 = [1,4,2,3]
vert_order3 = [1,3,4,2]
vert_order4 = [1,2,3,4]

@doc """
    tetface_areas(tet)

Calculate the area of the faces of a tetrahedron

# Arguments
- `tet::AbstractMatrix{<:Real}`: the points at the corners of the tetrahedron as
    the columns of a matrix.

# Returns
- `areas::Vector{<:Real}`: the areas of the faces of the tetrahedron. The order
    of the faces is determined by `face_ind`.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: tetface_areas
tet = [0 1 0 0; 0 0 1 0; 0 0 0 1]
tetface_areas(tet)
# output
4-element Vector{Float64}:
 0.8660254037844386
 0.5
 0.5
 0.5
```
"""
function tetface_areas(tet::AbstractMatrix{<:Real})::Vector{<:Real}
    areas = zeros(4)
    for (i,j)=enumerate(face_ind)
        areas[i] = norm(cross(tet[:,j[2]] - tet[:,j[1]], tet[:,j[3]] - tet[:,j[1]]))/2
    end
    areas
end

@doc """
    simpson3D(bezpts,quantity;num_slices,values,gauss,split,corner,atol,rtol)

Calculate the volume or hypervolume beneath a quadratic within a tetrahedron.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the quadratic polynomial over a 
    tetrahedron.
- `num_slices::Integer`: the number of slices of teterahedron parallel to one of the faces 
    of the tetrahedron.
- `quantity::String`: whether to calculate the "area" or "volume" of each slice.
- `values::Bool`: if true, return the areas or volumes of each of the slices.
- `gauss::Bool=true`: using Gaussian quadrature points if true.
- `atol::Real=def_atol`: an absolute tolerance.
- `split::Bool=true`: if true, split the integration interval where the slices are
    tangent to the quadratic surface.
- `corner::Union{Nothing,Integer}`: if provided, sliced approach the provided corner. 

# Returns
- The areas or volumes of slices of the tetrahedron or the volume or hypervolume
  of a polynomial within the tetrahedron. May instead be the values of curve being 
  integrated if `values` is true.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: simpson3D
spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0; 
    0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0]
coeffs = [-1/10, -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
bezpts= [spts; coeffs']
simpson3D(bezpts,"area",num_slices=10) ≈ 0.01528831567698499

# output
true
```
"""
function simpson3D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices, 
    values::Bool=false, gauss::Bool=true, split::Bool=true, corner::Union{Nothing,Integer}=nothing,
    atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(tetrahedron)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 3; deg = 2; cind = simplex_cornerpts(dim,deg); tetrahedron = bezpts[1:end-1,cind]
     # Area of faces
    if corner == nothing
        face_areas = tetface_areas(tetrahedron)
        p = findmax(face_areas)[2]
    else
        p = corner
    end 
    # Calculate the shortest distance from a plane at the opposite face to the
    # opposite corner
    face = tetrahedron[:,face_ind[p]]
    corner = tetrahedron[:,corner_ind[p]]
    n = cross(face[:,2] - face[:,1],face[:,3]-face[:,1])
    n = n ./ norm(n)
    d = abs(dot(corner - face[:,1],n))
 
    # Reorder coefficients and vertices
    coeff_order = @eval $(Symbol("coeff_order"*string(p)))
    slice_order = @eval $(Symbol("slice_order"*string(p)))
    vert_order = @eval $(Symbol("vert_order"*string(p)))

    if !split
        intervals = [0,1]
    else
        tpts = quadslice_tanpt(coeffs[coeff_order])[vert_order,:]
        if insimplex(tpts[:,1],atol=atol)
            if insimplex(tpts[:,2],atol=atol)
                intervals = [0; tpts[p,:]; 1]
            else
                intervals = [0,tpts[p,1],1]
            end
        elseif insimplex(tpts[:,2],atol=atol) 
            intervals = [0,tpts[p,2],1]
        else
            intervals = [0,1]
        end
    end
     
    interval_lengths = d*diff(intervals)
    interval_divs = [x < 3 ? 3 : mod(x,2) == 0 ? x+1 : x for x=round.(Int,num_slices*interval_lengths)]
    integral = 0; integral_vals = []; its = []
    bpts2D = sample_simplex(2,2)
    for j = 1:length(interval_divs)
        if gauss
            x,w = gausslegendre(interval_divs[j])
            w = w ./ 2
            it = (intervals[j+1] - intervals[j])*(x ./ 2) .+ (intervals[j] + intervals[j+1])/2
        else
            it = range(intervals[j],stop=intervals[j+1],length=interval_divs[j])
        end
        intvals = zeros(length(it))

        # No need to consider the end points; they are always zero
        for (i,t) in enumerate(it)
            bpts = reduce(hcat,[[(1-t),0,0,t],
            [(1-t)/2,(1-t)/2,0,t],
            [0,(1-t),0,t],
            [(1-t)/2,0,(1-t)/2,t],
            [0,(1-t)/2,(1-t)/2,t],
            [0,0,1-t,t]])
            bpts = bpts[slice_order,:]
            pts = barytocart(bpts,tetrahedron)
            vals = eval_poly(bpts,coeffs,dim,deg)
            coeffs2D = getpoly_coeffs(vals,bpts2D,2,2)
            intvals[i] = quad_area_volume([pts; coeffs2D'],quantity)
        end
        if intvals[end] === NaN
            intvals[end] = 0
        end

        if values
            integral_vals = [integral_vals; intvals]
            its = [its; it]
            continue
        end

        if gauss
            integral += interval_lengths[j]*dot(w,intvals)
        else
            integral += simpson(intvals,interval_lengths[j])
        end
    end

    if values
        its,integral_vals
    else
        integral
    end
end

@doc """
    quadslice_tanpt(coeffs,atol,rtol)

Calculate the points where a quadratic surface in tangent to a slice of a tetrahedron.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic.
- `atol::Real=def_atol`: an absolute tolerance.
- `rtol::Real=def_rtol`: a relative tolerance.

# Returns
- `bpts::Matrix{Real}`: the points where slices of a tetrahedron parallel to a face
    of the tetrahedron are tangent to the quadratic. The points are in Barycentric
    coordinates.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration: quadslice_tanpt
coeffs = [-1/10., -1/10, 9/10, -1/10, -1/10, 9/10, -1/10, -1/10, -1/10, 9/10]
quadslice_tanpt(coeffs)
# output
4×2 Matrix{Float64}:
  1.31623   0.683772
  0.0       0.0
  0.0       0.0
 -0.316228  0.316228
```
"""
function quadslice_tanpt(coeffs::AbstractVector{<:Real}; atol::Real=def_atol,
    rtol::Real=def_rtol)
     
    # s^2, 2 s t, t^2, 2 s u, 2 t u, u^2, 2 s v, 2 t v, 2 u v, v^2
    (c2000,c1100,c0200,c1010,c0110,c0020,c1001,c0101,c0011,c0002) = coeffs

    sa = ((c0110-c0200-c1010+c1100)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    sb = -((2*(c0110-c0200-c1010+c1100)*(c0002*c0110^2+
        c0101*c0110*c1001-c0110^2*c1001-c0101^2*c1010-
        c0002*c0110*c1010+c0101*c0110*c1010+c0002*c0200*c1010+
        c0011^2*(c0200-c1100)-c0002*c0110*c1100+
        c0020*(c0101^2+c0200*c1001+c0002*(-c0200+c1100)-
        c0101*(c1001+c1100))+c0011*(-c0200*(c1001+c1010)+c0110*(c1001+c1100)+
        c0101*(-2*c0110+c1010+c1100)))*(c0110^2+(c1010-
        c1100)^2+c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))))

    sc = (c0110-c0200-c1010+c1100)*(c0002*c0110^4+2*c0110^3*c1001^2-
        c0110^2*c0200*c1001^2-2*c0002*c0110^3*c1010+
        2*c0002*c0110^2*c0200*c1010-4*c0101*c0110^2*c1001*c1010+
        2*c0101*c0110*c0200*c1001*c1010+2*c0101^2*c0110*c1010^2+
        c0002*c0110^2*c1010^2-c0101^2*c0200*c1010^2-
        2*c0002*c0110*c0200*c1010^2+c0002*c0200^2*c1010^2-
        2*c0002*c0110*(c0110^2-c0110*c1010+c0200*c1010)*c1100+
        c0002*c0110^2*c1100^2+c0020^2*(c0200*c1001^2-c0101^2*(c0200-2*c1100)+
        c0002*(c0200-c1100)^2-2*c0101*c1001*c1100)+
        c0011^2*(c0110^2*c0200+c0200*c1010*(2*c0200+c1010-2*c1100)+
        2*c0110*(c1100^2-c0200*(c1010+c1100)))+
        c0020*(-c0110^2*c1001^2-c0200*(c0011^2*c0200+(2*c0110-c0200)*c1001^2+
        2*c0011*c1001*c1010)+2*c0011*(c0011*c0200+c0110*c1001)*c1100-c0011^2*c1100^2+
        c0101^2*(c0110^2+2*c0200*c1010+c1100*(-2*c1010+c1100)-
        2*c0110*(c1010+c1100))-2*c0002*(c0200-c1100)*(c0110^2+c0200*c1010-
        c0110*(c1010+c1100))+2*c0101*(c0011*c0110*(c0200-2*c1100)-c0200*c1001*c1100+
        c0011*c1010*c1100+c0110*c1001*(c1010+2*c1100)))-
        2*c0011*((2*c0110-c0200)*c1001*(-c0200*c1010+c0110*c1100)+
        c0101*(c0110^3-c0200*c1010*c1100-2*c0110^2*(c1010+c1100)+
        c0110*(2*c0200*c1010+c1010^2+c1100^2))))

    ta = ((c0110-c1010-c1100+c2000)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+
        2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    tb = -((2*(c0110-c1010-c1100+c2000)*(c0110^2+(c1010-c1100)^2+
        c0200*(2*c1010-c2000)-2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(-c0110*c1001^2-
        c0002*c0110*c1010+c0101*c1001*c1010+c0110*c1001*c1010+
        c0002*c1010^2-c0101*c1010^2-c0002*c1010*c1100+
        c0002*c0110*c2000+c0011^2*(-c1100+c2000)+
        c0020*(-c1001*(c0101-c1001+c1100)+c0002*(c1100-c2000)+
        c0101*c2000)+c0011*(c0110*c1001+c0101*c1010-2*c1001*c1010+
        c1001*c1100+c1010*c1100-(c0101+c0110)*c2000))))

    tc = ((c0110-c1010-c1100+
        c2000)*(c1010*(-2*c0011*c1001*(c0110-c1010)^2+
        c0110^2*(2*c1001^2+c0002*c1010)-
        2*c0110*c1010*(2*c0101*c1001+c0002*(c1010-c1100))+
        c1010*(2*c0101^2*c1010+c0002*(c1010-c1100)^2)+
        4*c0011*(-c0101+c1001)*c1010*c1100+2*c0011^2*c1100^2-
        2*c0011*c1001*c1100^2)+(-c0101^2*c1010^2-
        c0110^2*(c1001^2+2*c0002*c1010)+
        2*c0110*c1010*(c0101*c1001+c0002*c1010-c0002*c1100)+
        c0011^2*((c0110-c1010)^2-2*(c0110+c1010)*c1100)+
        2*c0011*(c0101*c1010*(2*c0110+c1100)+
        c0110*c1001*(-2*c1010+c1100)))*c2000+
        c0110*(2*c0011*(c0011-c0101)+c0002*c0110)*c2000^2+
        c0020^2*(-2*c0101*c1001*c1100+c0002*(c1100-c2000)^2+
        c1001^2*(2*c1100-c2000)+c0101^2*c2000)+
        c0020*(-2*c0110*c1001^2*c1010+c1001^2*c1010^2+
        2*c0011*c0110*c1001*c1100-2*c0110*c1001^2*c1100-
        2*c0002*c0110*c1010*c1100-4*c0011*c1001*c1010*c1100-
        2*c1001^2*c1010*c1100+2*c0002*c1010^2*c1100-
        c0011^2*c1100^2+c1001^2*c1100^2-2*c0002*c1010*c1100^2+
        2*c0101*c1010*(c0110*c1001+(c0011+2*c1001)*c1100)-
        2*c0101*(c0011*c0110+c1001*c1100)*c2000+
        2*(c0011*c1001*c1010+c0011^2*c1100+
        c0002*c1010*(-c1010+c1100)+c0110*(c1001^2+
        c0002*(c1010+c1100)))*c2000-(c0011^2+2*c0002*c0110)*c2000^2+
        c0101^2*(-c1010^2-2*c1010*c2000+c2000^2))))

    ua = ((c0110^2+(c1010-c1100)^2+c0200*(2*c1010-c2000)-
        2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(c0002*c0110^2+
        2*c0101*c0110*c1001-2*c0110^2*c1001-2*c0110*c1001^2+
        c0200*c1001^2-2*c0101^2*c1010-2*c0002*c0110*c1010+
        2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+
        2*c0110*c1001*c1100-2*c0002*c1010*c1100+
        2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+
        c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-
        c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000))))

    ub = -((2*(c0110^2+(c1010-c1100)^2+c0200*(2*c1010-c2000)-
        2*c0110*(c1010+c1100-c2000)-
        c0020*(c0200-2*c1100+c2000))*(-c0110*c1001^2+
        c0200*c1001^2+c0002*c0200*c1010-c0200*c1001*c1010-
        c0002*c0110*c1100+c0110*c1001*c1100-c0002*c1010*c1100+
        c0011*(c1001-c1100)*c1100+c0002*c1100^2+
        c0002*(c0110-c0200)*c2000+c0011*c0200*(-c1001+c2000)+
        c0101^2*(-c1010+c2000)+c0101*(c0110*c1001+c1001*c1010+c0011*c1100-
        2*c1001*c1100+c1010*c1100-(c0011+c0110)*c2000))))

    uc = (-2*c0110*c0200*c1001^2*c1010+2*c0200^2*c1001^2*c1010+
        c0002*c0200^2*c1010^2+c0200*c1001^2*c1010^2+
        2*c0110^2*c1001^2*c1100-2*c0110*c0200*c1001^2*c1100-
        2*c0002*c0110*c0200*c1010*c1100-2*c0200*c1001^2*c1010*c1100-
        2*c0002*c0200*c1010^2*c1100+c0002*c0110^2*c1100^2+
        c0200*c1001^2*c1100^2+2*c0002*c0110*c1010*c1100^2+
        2*c0002*c0200*c1010*c1100^2+c0002*c1010^2*c1100^2-
        2*c0002*c0110*c1100^3-2*c0002*c1010*c1100^3+
        c0002*c1100^4-(c0110-c0200)*(-c0200*(c1001^2+2*c0002*c1010)+
        2*c0002*(c1010-c1100)*c1100+
        c0110*(c1001^2+2*c0002*c1100))*c2000+
        c0002*(c0110-c0200)^2*c2000^2-
        2*c0011*c1001*(c0200*c1010-c0110*c1100)*(c0200-2*c1100+
        c2000)+c0011^2*(c0200-2*c1100+c2000)*(-c1100^2+c0200*c2000)+
        c0101^2*(2*c1010^2*c1100-c0200*(c1010-c2000)^2-
        2*c1010*(c0110+c1100)*c2000+
        c2000*((c0110-c1100)^2+2*c0110*c2000))+
        2*c0101*(-c0110^2*c1001*c1100-
        c1001*c1100*((c1010-c1100)^2+c0200*(2*c1010-c2000))+
        c0011*c1010*c1100*(c0200-2*c1100+c2000)+
        c0110*(2*c1001*c1100*(c1100-c2000)+c1001*c1010*c2000+
        c0011*(2*c1100-c2000)*c2000+c0200*(c1001*c1010-c0011*c2000))))

    va = (c0002*c0110^2+2*c0101*c0110*c1001-2*c0110^2*c1001-
        2*c0110*c1001^2+c0200*c1001^2-2*c0101^2*c1010-
        2*c0002*c0110*c1010+2*c0101*c0110*c1010+2*c0002*c0200*c1010+
        2*c0101*c1001*c1010+2*c0110*c1001*c1010-2*c0200*c1001*c1010+
        c0002*c1010^2-2*c0101*c1010^2+c0200*c1010^2-
        2*c0002*c0110*c1100-2*c0101*c1001*c1100+2*c0110*c1001*c1100-
        2*c0002*c1010*c1100+2*c0101*c1010*c1100-2*c0110*c1010*c1100+
        c0002*c1100^2+((c0101-c0110)^2+
        c0002*(2*c0110-c0200))*c2000+
        c0011^2*(c0200-2*c1100+c2000)+
        c0020*(c0101^2+(c1001-c1100)^2+c0200*(2*c1001-c2000)-
        2*c0101*(c1001+c1100-c2000)-
        c0002*(c0200-2*c1100+c2000))-
        2*c0011*(c0200*c1001+c0200*c1010+c1001*c1010-c1001*c1100-
        c1010*c1100+c1100^2-c0110*(c1001+c1100-c2000)-
        c0200*c2000+c0101*(c0110-c1010-c1100+c2000)))

    vb = -((2*(-c0011*c0200*c1010-c0200*c1001*c1010-c0101*c1010^2+
        c0200*c1010^2+c0011*c1010*c1100+c0101*c1010*c1100-
        c0011*c1100^2+c0011*c0200*c2000+c0110^2*(-c1001+c2000)+
        c0020*(-(c0101+c1001-c1100)*c1100+c0200*(c1001-c2000)+
        c0101*c2000)+c0110*(c0101*c1010+c1001*c1010+c0011*c1100+c1001*c1100-
        2*c1010*c1100-(c0011+c0101)*c2000))))

    vc = (-2*c0110*c1010*c1100+c0020*c1100^2+c0110^2*c2000+
        c0200*(c1010^2-c0020*c2000))

    abcs = [[sa,sb,sc],[ta,tb,tc],[ua,ub,uc],[va,vb,vc]]
    stuv = [[0.,0.],[0.,0.],[0.,0.],[0.,0.]]
    for (i,abc) in enumerate(abcs)
        a,b,c=abc
        sol = solve_quadratic(abc[1],abc[2],abc[3],atol=atol)
        if length(sol) == 0
            sol = [0,0]
        elseif length(sol) == 1
            sol = [sol[1],sol[1]]
        end
        stuv[i] = sol
    end
     
    s,t,u,v = stuv
    bpts = zeros(4,2); b1 = zeros(4); b2 = zeros(4)
    for i=1:2, j=1:2, k=1:2, l=1:2
        b1 = [s[l],t[k],u[j],v[i]]
        if isapprox(sum(b1),1, atol=atol,rtol=rtol)
            b2 = [s[mod1(l+1,2)],t[mod1(k+1,2)],u[mod1(j+1,2)],v[mod1(i+1,2)]]
            if isapprox(sum(b2),1,atol=atol,rtol=rtol)
                bpts = [b1 b2]
                break
            end
        end
    end
    bpts
end

@doc """
    length_area1D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the area or length of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the bezier points of the univariate polynomial.
- `quantity::String`: the quantity calculate. "area" gives the length of the domain
    where the polynomial is less than zero, and "volume" gives the area (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the polynomial at the quadrature points when true.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The length of the domain or the area below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
coeffs = [1.1,2.2,3.3,4.4]
spts = [-6/3,-1/3,4/3,9/3]
bezpts = [spts'; coeffs']
num_slices = 10
quantity = "area"
length_area1D(bezpts,quantity,num_slices=num_slices)
# output
0.0
```
"""
function length_area1D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices,
    gauss::Bool=true, values::Bool=false, atol::Real=def_atol)

    interval = [bezpts[1,1],bezpts[1,end]]
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return abs(interval[2] - interval[1])
        elseif quantity == "volume"
            return mean(coeffs)*abs(interval[2] - interval[1])
        end 
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 1; deg = length(coeffs) - 1
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    bpts = reduce(hcat,[[i,1-i] for i=it])
    vals = eval_poly(bpts,coeffs,dim,deg)
    if quantity == "area"
        replace!(x -> (x > 0 || isapprox(x,0,atol=atol)) ? 0 : 1, vals)
    elseif quantity == "volume"
        replace!(x -> (x > 0 || isapprox(x,0,atol=atol)) ? 0 : x, vals)
    else
        error("Invalid quantity")
    end    
    if values
        return it,vals
    end

    if gauss
        abs(interval[2] - interval[1])*dot(w,vals)
    else
        simpson(vals,abs(interval[2]-interval[1]))
    end
end

@doc """
    area_volume2D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the area or length of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the bivariate polynomial over a 
    triangle.
- `quantity::String`: the quantity calculate. "area" gives the area of the domain
    where the polynomial is less than zero, and "volume" gives the volume (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of slices. Each slices has this many quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the integral of the polynomial at the quadrature points when true.
- `edge_ind::Union{Nothing,Integer}=nothing`: if provided, slices of triangle start
    from this edge.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The area of the domain or the volume below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
spts = [0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
coeffs = [0.1, 0.2, 0.4, 0.4, 0.3, 0.3]
bezpts = [spts; coeffs']
area_volume2D(bezpts,"area")
# output
0.0
```
"""
function area_volume2D(bezpts::Matrix{<:Real}, quantity::String; num_slices::Integer=def_num_slices,
    gauss::Bool=true, values::Bool=false, edge_ind::Union{Nothing,Integer}=nothing, atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(triangle)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 2; deg = 0; n = 0 
    while n != length(coeffs)
        deg += 1
        n = ntripts(deg)
    end
    cind = simplex_cornerpts(dim,deg)
    triangle = bezpts[1:end-1,cind]
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    if edge_ind == nothing
        # Move from the edge of the triangle of the largest length to the opposite corner
        lengths = [norm(triangle[:,mod1(i,3)] - triangle[:,mod1(i+1,3)]) for i=1:3]
        corner_midpoint_lens = [norm([mean(triangle[:,[mod1(i,3),mod1(i+1,3)]],dims=2)...] -
                triangle[:,mod1(i+2,3)]) for i=1:3]
        edge_ind = findmax(lengths)[2]
    end
    if edge_ind == 1
       order = [2,3,1]  
    elseif edge_ind == 2
        order = [1,2,3]
    else
        order = [3,1,2]
    end 
    ibpts= sample_simplex(1,deg)
    intvals = zeros(length(it))
    for (i,t) in enumerate(it)
        bpt = [t,(1-t)/2,(1-t)/2][order]
        e1bpt = [t,0,1-t][order]
        e2bpt = [t,1-t,0][order]
        bpts = [e1bpt e2bpt]
        pts = barytocart(bpts,triangle)
        polypts = reduce(hcat,[(1-t)*pts[:,1] + t*pts[:,2] for t=0:1/(deg):1])
        polybpts = carttobary(polypts,triangle)
        vals = eval_poly(polybpts,coeffs,dim,deg)
        l = norm(pts[:,1] - pts[:,end])
        if deg < 3
            bezcoeffs = get_1Dquad_coeffs(vals)
        else
            bezcoeffs = getpoly_coeffs(vals,ibpts,1,deg)
        end
        intvals[i] = length_area1D([barytocart(ibpts,Matrix([0,l]')); bezcoeffs'], quantity, 
            num_slices=num_slices, gauss=gauss, atol=atol)
    end
    
    if values
        return it,intvals
    end
    edge = triangle[:,[edge_ind,mod1(edge_ind+1,3)]]
    opp_corner = triangle[:,mod1(edge_ind+2,3)]
    d = linept_dist(edge,opp_corner)
    if gauss
        d*dot(w,intvals)
    else
        simpson(intvals,d)
    end
end

@doc """
    volume_hypvol3D(bezpts,quantity;num_slices,gauss,values,atol)

Calculate the volume or hyper volume of a polynomial where it is less than zero.

# Arguments
- `bezpts::Matrix{<:Real}`: the Bezier points of the trivariate polynomial over a
    tetrahedron.
- `quantity::String`: the quantity calculate. "area" gives the area of the domain
    where the polynomial is less than zero, and "volume" gives the volume (negative) 
    beneath the polynomial where the polynomial is less than zero.
- `num_slices::Integer`: the number of slices. Each slices has this many quadrature points.
- `gauss::Bool=true`: if true, use Gaussian quadrature points.
- `values::Bool=false`: returns the quadrature points (between 0 and 1) and the 
    values of the integral of the polynomial at the quadrature points when true.
- `corner::Union{Nothing,Integer}=nothing`: if provided, slices of tetrahedron approach this corner.
- `atol::Real=def_atol`: an absolute tolerance for floating-point comparisons.

# Returns
- The volume of the domain or the hyper volume below the polynomial where the polynomial 
    is less than zero.

# Examples
```jldoctest
using Pebsi.QuadraticIntegration
spts = [0.0 0.5 1.0 0.0 0.5 0.0 0.0 0.5 0.0 0.0; 0.0 0.0 0.0 0.5 0.5 1.0 0.0 0.0 0.5 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.5 0.5 0.5 1.0];
coeffs = collect(1:10)
bezpts = [spts; coeffs']
quantity = "area"
volume_hypvol3D(bezpts,quantity)
# output
0.0
```
"""
function volume_hypvol3D(bezpts::Matrix{<:Real}, quantity::String;
    num_slices::Integer=def_num_slices, gauss::Bool=true, values::Bool=false, 
    corner::Union{Nothing,Integer}=nothing, atol::Real=def_atol)
    coeffs = bezpts[end,:]
    # All the coefficients are well below zero.
    if all((coeffs .< 0) .& isapprox.(coeffs,0,atol=atol))
        if quantity == "area"
            return simplex_size(tetrahedron)
        elseif quantity == "volume"
            return mean(coeffs)*simplex_size(tetrahedron)
        end    
    # All the coefficients are well above zero.
    elseif all((coeffs .> 0) .& isapprox.(coeffs,0,atol=atol))
       return 0.0 
    end
    dim = 3; deg = 0; n = 0
    while n != length(coeffs)
        deg += 1
        n = ntetpts(deg)
    end
    cind = simplex_cornerpts(dim,deg)
    tetrahedron = bezpts[1:end-1,cind] 
    if gauss
        x,w = gausslegendre(num_slices)
        w = w ./ 2
        it = (x ./ 2) .+ 1/2
    else
        if iseven(num_slices) num_slices += 1 end
        it = range(0,stop=1,length=num_slices)
    end
    
    # Area of faces
    if corner == nothing
        face_areas = tetface_areas(tetrahedron)
        p = findmax(face_areas)[2]
    else
        p = corner
    end
   
    # Calculate the shortest distance from a plane at the opposite face to the
    # opposite corner
    face = tetrahedron[:,face_ind[p]]
    corner = tetrahedron[:,corner_ind[p]]
    n = cross(face[:,2] - face[:,1],face[:,3]-face[:,1])
    n = n ./ norm(n)
    d = abs(dot(corner - face[:,1],n))
    
    # Reorder coefficients and vertices
    slice_order = @eval $(Symbol("slice_order"*string(p)))
    intvals = zeros(length(it))
    bpts2D = sample_simplex(2,deg)
    # No need to consider the end points; they are always zero
    for (i,t) in enumerate(it)
        bpts = reduce(hcat,[[(1-t),0,0,t],[0,(1-t),0,t],[0,0,1-t,t]])
        bpts = bpts[slice_order,:]
        pts = barytocart(bpts,tetrahedron)
        polypts = barytocart(bpts2D,pts)
        polybpts = carttobary(polypts,tetrahedron)
        vals = eval_poly(polybpts,coeffs,dim,deg)
        coeffs2D = getpoly_coeffs(vals,bpts2D,2,deg)
        triangle = Utilities.mapto_xyplane(pts)
        intvals[i] = area_volume2D([barytocart(bpts2D,triangle); coeffs2D'],quantity,num_slices=num_slices,
            gauss=gauss,atol=atol)
    end
    if values return it,intvals end
    if gauss
        d*dot(w,intvals)
    else
        simpson(intvals,d)
    end        
end
