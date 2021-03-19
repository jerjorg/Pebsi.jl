module QuadraticIntegration

include("Polynomials.jl")

import .Polynomials: sample_simplex,eval_poly,getpoly_coeffs,barytocart,carttobary


import LinearAlgebra: cross
import MiniQhull: delaunay

"""
    order_vertices(vertices)

Put the vertices of a triangle (columns of an array) in counterclockwise order.
"""
function order_vertices!(vertices::AbstractArray{<:Real,2})
    for i=1:3
        j = mod1(i+1,3)
        k = mod1(i+2,3)
        (v1,v2,v3) = [vertices[:,l] for l=[i,j,k]]
        if cross(vcat(v2-v1,0),vcat(v3-v1,0))[end] < 0
            v = vertices[:,k]
            vertices[:,k] = vertices[:,j]
            vertices[:,j] = v
        end 
    end
    vertices
end


"""
    quadval_vertex(bezcoeffs)

Calculate the value of a quadratic curve at its vertex.

# Arguments
- `bezcoeffs::AbstractArray{<:Real,1}`: the quadratic polynomial coefficients.
"""
function quadval_vertex(bezcoeffs::AbstractArray{<:Real,1})
    (a,b,c) = bezcoeffs
    (-b^2+a*c)/(a-2b+c)
end

"""
    edge_intersects(bezpts,atol)

Calculate where a quadratic curve is equal to zero within [0,1).

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points (columns of an array).
- `atol::Real=1e-12`: absolute tolerance for comparisons of floating point 
    numbers with zero.

# Returns
- `::AbstractArray{<:Real,1}`: an array of up to two intersections as real numbers.

# Examples
import Pebsi.QuadraticIntegration: edge_intersects
coeffs = [1,0,-1]
bezpts = vcat(cartpts,coeffs')
edge_intersects(bezpts) == [0.5]
# output
1-element Array{Float64,1}:
 0.5
"""
function edge_intersects(bezpts::AbstractArray{<:Real,2};
    atol::Real=1e-12)::AbstractArray{<:Real,1}
    
    # Cases where the curve is above zero, below zero, or at zero.
    coeffs = bezpts[end,:]
    if all(isapprox.(coeffs,0,atol=atol))
        return Array{Float64}([])
    elseif !any(isapprox.(coeffs,0,atol=atol)) && all(coeffs .> 0)
        println("no intersections 1")
        return Array{Float64}([])
    elseif !any(isapprox.(coeffs,0,atol=atol)) && all(coeffs .< 0)
        println("no intersections 2")
        return Array{Float64}([])
    end
    
    # Put the polynomial in a form where cases are easier to handle:
    # α + βx + γx² == 0
    (a,b,c)=coeffs
    α = a
    β = -2a+2b
    γ = a-2b+c
    
    # Quadratic curve entirely above or below zero.
    v = quadval_vertex(coeffs)
    if all([γ,v].>=0) || all([γ,v].<=0) && abs(v) != Inf
        return Array{Float64}([])
    end
    
    if γ==0 && β==0
        x = Array{Float64}([])
    elseif γ==0
        x = [-α/β]
    else
        arg = β^2-4α*γ
        if isapprox(arg,0,atol=atol)
            # There are two solutions at the same point if arg == 0 but we only
            # keep one of them.
            x = [-β/(2γ)]
        elseif arg < 0
            # Ignore solutions with imaginary components.
            x = Array{Float64}[]
        else
            x = [(-β-sqrt(arg))/(2γ),(-β+sqrt(arg))/(2γ)]
        end
    end

    # Only keep non-complex intersections between [0,1).
    filter(y -> (y>0 || isapprox(y,0,atol=atol)) && y<1 && 
        !isapprox(y,1,atol=atol),x) |> sort
end


"""
The locations of quadratic Bezier points at the corners of the triangle in
counterclockwise order.
"""
corner_indices = [1,3,6]

"""
The locations of quadratic Bezier points along each edge of the triangle in
counterclockwise order.
"""
edge_indices=[[1,2,3],[3,5,6],[6,4,1]]

@doc """
    simplex_intersects(bezpts,atol)

Calculate the location where a level curve intersects a triangle.

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points of the quadratic, Bezier
    surface.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `intersects::Array`: the intersections organized by edge in a 1D array. Each 
    element of the array is a 2D array where the columns are the Cartesian
    coordinates of intersections.

# Examples
```jldoctest
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 -1.0 -2.0 1.0 0.0 2.0]
simplex_intersects(bezpts)
# output
3-element Array{Array,1}:
 [-1.0; 0.0]
 [0.5; 0.5]
 Any[]
```
"""
function simplex_intersects(bezpts::AbstractArray{<:Real,2},
    atol::Real=1e-12)::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = edge_intersects(edge_bezpts,atol=atol)
        if edge_ints != []
            intersects[i] = reduce(hcat,[edge_bezpts[1:2,1] .+ 
                i*(edge_bezpts[1:2,end] .- edge_bezpts[1:2,1]) for i=edge_ints])
        end
    end
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects == 1
        Array{Array,1}([[],[],[]])
    else
        intersects
    end
end

"""
    conicsection(coeffs,atol)

Classify the conic section of a level curve of a quadratic surface.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the quadratic polynomial.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `::String`: the type of the conic section.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: conicsection
coeffs = [0.36, -1.64, 0.36, -0.64, -0.64, 0.36]
# output
"elipse"
```
"""
function conicsection(coeffs::AbstractArray{<:Real,1},
    atol::Real=1e-12)::String
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
    a = z₀₀₂ - 2z₁₀₁ + z₂₀₀
    b = 2z₀₀₂ - 2z₀₁₁ - 2z₁₀₁ + 2z₁₁₀
    c = z₀₀₂ - 2z₀₁₁ + z₀₂₀
    d = b^2 - 4*a*c
    m = -8*(-2*z₀₁₁*z₁₀₁*z₁₁₀+z₀₀₂*z₁₁₀^2+z₀₁₁^2*z₂₀₀+z₀₂₀*(z₁₀₁^2-z₀₀₂*z₂₀₀))

    @show (a,b,c,d,m)

    if all(isapprox.([a,b,c],0,atol=atol))
        "line"
    elseif isapprox(m,0,atol=atol)
        if isapprox(d,0,atol=atol)
            "parallel lines"
        elseif d > 0
            "rectangular hyperbola"
        else # d < 0
            "point"
        end
    else
        if isapprox(d,0,atol=atol)
            "parabola"
        elseif d < 0
            "elipse"
        else # d > 0
            "hyperbola"
        end
    end
end

"""
    saddlepoint(coeffs)

Calculate the saddle point of a quadratic Bezier surface.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the quadratic polynomial.
- `atol::Real=1e-12`: absolute tolerance.

# Returns
- `::AbstractArray{<:Real,1}`: the coordinates of the saddle point in Barycentric coordinates.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: saddlepoint
coeffs = [0.36, -1.64, 0.36, -0.64, -0.64, 0.36]
saddlepoint(coeffs)
# output
3-element Array{Float64,1}:
 0.5000000000000001
 4.163336342344338e-17
 0.5000000000000001
```
"""
function saddlepoint(coeffs::AbstractArray{<:Real,1},
    atol::Real=1e-12)::AbstractArray{<:Real,1}
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
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
    simplex_size(simplex)

Calculate the size of the region within a simplex.

# Arguments
- `simplex::AbstractArray{<:Real,2}`: the vertices of the simplex as columns of 
    an array.

# Returns
- `::Real`: the size of the region within the simplex. For example, the area
    within a triangle in 2D.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: simplex_size
simplex = [0 0 1; 0 1 0]
simplex_size(simplex)
# output
0.5
```
"""
function simplex_size(simplex::AbstractArray{<:Real,2})::Real
    abs(1/factorial(size(simplex,1))*det(vcat(simplex,ones(1,size(simplex,2)))))
end

@doc """
    shadow_size(coeff,simplex,val;rtol,atol)

Calculate the size of the shadow of a linear or quadratic Bezier triangle.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the Bezier triangle.
- `simplex::AbstractArray{<:Real,2}`: the domain of the Bezier triangle.
- `val::Real`: the value of a cutting plane.
- `rtol::Real=sqrt(eps(float(maximum(coeffs))))`: a relative tolerance for 
    floating point comparisons.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `::Real`: the size of the shadow of the Bezier triangle within `simplex` and 
    below a cutting plane of height `val`.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: shadow_size
coeffs = [0.4, 0.5, 0.3, -0.2, -0.1, -0.3, 0.7, -0.6, 0.9, -0.7]
simplex = [0.0 0.5 0.5 0.0; 1.0 1.0 0.0 0.0; 0.0 0.0 0.0 1.0]
val = 0.9
shadow_size(coeffs,simplex,val)
# output
0.08333333333333333
```
"""
function shadow_size(coeffs::AbstractArray{<:Real,1},
    simplex::AbstractArray{<:Real,2},val::Real;
    rtol::Real=sqrt(eps(float(maximum(coeffs)))),
    atol::Real=1e-9)::Real
    
    if minimum(coeffs) > val|| isapprox(minimum(coeffs),val,rtol=rtol,atol=atol)
        0
    elseif maximum(coeffs) < val || isapprox(maximum(coeffs),val,rtol=rtol,atol=atol)
        simplex_size(simplex)
    else
        1e10
    end
end

@doc """
    bezsimplex_size(coeff,simplex,val;rtol,atol)

Calculate the size of the shadow of a linear or quadratic Bezier triangle.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the Bezier triangle.
- `simplex::AbstractArray{<:Real,2}`: the domain of the Bezier triangle.
- `val::Real`: the value of a cutting plane.
- `rtol::Real=sqrt(eps(float(maximum(coeffs))))`: a relative tolerance for 
    floating point comparisons.
- `atol::Real=1e-9`: an absolute tolerance for floating point comparisons.

# Returns
- `::Real`: the size of the shadow of the Bezier triangle within `simplex` and 
    below a cutting plane of height `val`.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: bezsimplex_size
coeffs = [0.4, 0.5, 0.3, -0.2, -0.2, -0.3]
simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
bezsimplex_size(coeffs,simplex,100)
# output
0.020833333333333332
```
"""
function bezsimplex_size(coeffs::AbstractArray{<:Real,1},
    simplex::AbstractArray{<:Real,2},val::Real;
    rtol::Real=sqrt(eps(float(maximum(coeffs)))),
    atol::Real=1e-9)::Real
    
    if maximum(coeffs) < val || isapprox(maximum(coeffs),val,rtol=rtol,atol=atol)
        simplex_size(simplex)*mean(coeffs)
    elseif minimum(coeffs) > val || isapprox(minimum(coeffs),val,rtol=rtol,atol=atol)
        0
    else
       1e10
    end
    
end

@doc """
    insimplex(bpt,atol)

Check if a point lie within a simplex (including the boundary).

# Arguments
- `bpt::AbstractArray{<:Real,2}`: a point in Barycentric coordinates.
- `atol::Real=1e-12`: an absolute tolerance.

# Returns
- `::Bool`: a boolean indicating if the point is within the simplex.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: insimplex
bpt = Array([0 1]')
insimplex(bpt)
# output
true
```
"""
function insimplex(bpt::AbstractArray{<:Real,1},atol::Real=1e-12)
    (isapprox(maximum(bpt),1,atol=atol) || maximum(bpt) < 1) &&
    (isapprox(minimum(bpt),0,atol=atol) || minimum(bpt) > 0) &&
    isapprox(sum(bpt),1,atol=atol)
end

@doc """
    insimplex(bpts,atol)

Check if an array of points in Barycentric coordinates lie within a simplex.

# Arguments
- `bpts::AbstractArray{<:Real,2}`: an arry of points in barycentric coordinates
    as columns of an array.
- `atol::Real=1e-12`: absolute tolerance.
"""
function insimplex(bpts::AbstractArray{<:Real,2},atol::Real=1e-12)
    all(mapslices(x->insimplex(x,atol),bpts,dims=1))
end

@doc """
    split_bezsurf₁(bezpts,atol)

Split a Bezier surface into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points of the quadratic surface.
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_triangle
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 0.0 1.0 -1.0 0.0]
split_bezsurf₁(bezpts)
# output
3-element Array{Array{Float64,2},1}:
 [0.0 0.5 … 0.0 -1.0; 0.6 0.3 … 0.0 0.0; 0.08000000000000002 -0.40000000000000013 … 1.0 0.0]
 [0.0 0.0 … -0.5 -1.0; 0.6 0.8 … 0.5 0.0; 0.08000000000000002 -2.7755575615628914e-17 … 1.0 0.0]
 [0.0 0.0 … 0.5 1.0; 0.6 0.8 … 0.5 0.0; 0.08000000000000002 -2.7755575615628914e-17 … -1.0 0.0]
```
"""
function split_bezsurf₁(bezpts::AbstractArray{<:Real,2},
    atol::Real=1e-9)::AbstractArray

    dim = 2
    deg = 2
    triangle = bezpts[1:2,[1,3,6]]
    coeffs = bezpts[end,:]
    simplex_bpts = sample_simplex(dim,deg)
    intersects = simplex_intersects(bezpts,atol)
    allintersects = reduce(hcat,[i for i=intersects if i!=[]])
    spt = saddlepoint(coeffs,atol)
    if insimplex(spt)
        allpts = [triangle barytocart(spt,triangle) allintersects]
    else
        allpts = [triangle allintersects]
    end

    tri_ind = delaunay(allpts)
    subtri = [allpts[:,tri_ind[:,i]] for i=1:size(tri_ind,2)]
    sub_pts = [barytocart(simplex_bpts,tri) for tri=subtri]
    sub_bpts = [carttobary(pts,triangle) for pts=sub_pts]
    sub_vals = [reduce(hcat, [eval_poly(sub_bpts[j][:,i],coeffs,dim,deg) 
        for i=1:6]) for j=1:length(subtri)]
    sub_coeffs = [getpoly_coeffs(v[:],simplex_bpts,dim,deg) for v=sub_vals]
    sub_bezpts = [[sub_pts[i]; sub_coeffs[i]'] for i=1:length(sub_coeffs)]
    sub_bezpts
end

@doc """
    split_bezsurf(bezpts,atol)

Split a Bezier surface into sub-Bezier surfaces with the Delaunay method.

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points of the quadratic surface.
- `atol::Real=1e-9`: absolute tolerance.

# Returns
- `sub_bezpts::AbstractArray`: the Bezier points of the sub-surfaces.

# Examples
```jldoctest
import Pebsi.QuadraticIntegration: split_triangle
bezpts = [-0.09385488270304788 0.12248162346376468 0.3388181296305772 0.09890589198180941 0.315242398148622 0.2916666666666667; 0.9061451172969521 0.7836634938331875 0.6611818703694228 0.6266836697595872 0.5042020462958225 0.34722222222222227; 0.0 7.949933953535975 3.9968028886505635e-15 8.042737134030771 -5.792491135426262 -11.720219017094017]
split_bezsurf₁(bezpts)
# output
2-element Array{Array{Float64,2},1}:
 [0.1291676795676943 0.23399290459913574 … 0.12248162346376468 -0.09385488270304788; 0.5828106204960847 0.6219962454327537 … 0.7836634938331875 0.9061451172969521; -5.329070518200751e-15 -4.5330445462060594e-15 … 7.9499339535359725 0.0]
 [0.1291676795676943 0.2104171731171805 … 0.315242398148622 0.3388181296305772; 0.5828106204960847 0.46501642135915344 … 0.5042020462958225 0.6611818703694228; -5.329070518200751e-15 -3.39004820851129 … -5.792491135426261 -1.1479627341393213e-15]
```
"""
function split_bezsurf(bezpts::AbstractArray{<:Real,2},atol=1e-12)::AbstractArray
    
    intersects = simplex_intersects(bezpts)
    num_intersects = sum([size(i,2) for i=intersects if i!=[]])
    if num_intersects <= 2
        return [bezpts]
    else
        sub_bezpts = split_bezsurf₁(bezpts)
        sub_intersects = [simplex_intersects(b) for b=sub_bezpts]
        num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
            size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
        while any(num_intersects .> 2)
            for i = length(num_intersects):-1:1
                if num_intersects[i] <= 2 continue end
                append!(sub_bezpts,split_bezsurf₁(sub_bezpts[i]))                
                deleteat!(sub_bezpts,i)
                sub_intersects = [simplex_intersects(b) for b=sub_bezpts]
                num_intersects = [sum([size(sub_intersects[i][j])[1] == 0 ? 0 : 
                    size(sub_intersects[i][j])[2] for j=1:3]) for i=1:length(sub_intersects)]
            end
        end
    end
    sub_bezpts
end

end # module