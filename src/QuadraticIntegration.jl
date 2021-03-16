module QuadraticIntegration

import LinearAlgebra: cross

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
    edge_intersects(bezpts)

Calculate where a quadratic curve is equal to zero within [0,1).

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points (columns of an array).

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
    atol=1e-12)::AbstractArray{<:Real,1}
    
    # Cases where the curve is above zero, below zero, or at zero.
    coeffs = bezpts[end,:]
    if all(isapprox.(coeffs,0,atol=atol))
        return Array{Float64}([])
    elseif !any(isapprox.(coeffs,0,atol=atol)) && all(coeffs .> 0)
        print("no intersections 1")
        return Array{Float64}([])
    elseif !any(isapprox.(coeffs,0,atol=atol)) && all(coeffs .< 0)
        print("no intersections 2")
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
    filter(y -> (y>0 || isapprox(y,0,atol=atol)) && y<1 ,x) |> sort
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
    simplex_intersects(bezpts)

Calculate the location where a level curve intersects a triangle.

# Arguments
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points of the quadratic, Bezier
    surface.

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
function simplex_intersects(bezpts::AbstractArray{<:Real,2})::Array
    intersects = Array{Array,1}([[],[],[]])
    for i=1:3
        edge_bezpts = bezpts[:,edge_indices[i]]
        edge_ints = edge_intersects(edge_bezpts)
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
    conicsection(coeffs)

Classify the conic section of a level curve of a quadratic surface.

# Arguments
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the quadratic polynomial.

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
function conicsection(coeffs::AbstractArray{<:Real,1})::String
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
    a = z₀₀₂ - 2z₁₀₁ + z₂₀₀
    b = 2z₀₀₂ - 2z₀₁₁ - 2z₁₀₁ + 2z₁₁₀
    c = z₀₀₂ - 2z₀₁₁ + z₀₂₀
    d = b^2 - 4*a*c
    m = -8*(-2*z₀₁₁*z₁₀₁*z₁₁₀+z₀₀₂*z₁₁₀^2+z₀₁₁^2*z₂₀₀+z₀₂₀*(z₁₀₁^2-z₀₀₂*z₂₀₀))

    @show (a,b,c,d,m)

    if a == 0 && b == 0 && c == 0
        "line"
    elseif m == 0
        if d == 0
            "parallel lines"
        elseif d > 0
            "rectangular hyperbola"
        else # d < 0
            "point"
        end
    else
        if d == 0
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
function saddlepoint(coeffs::AbstractArray{<:Real,1})::AbstractArray{<:Real,1}
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
    denom = z₀₁₁^2+(z₁₀₁-z₁₁₀)^2+z₀₂₀*(2z₁₀₁-z₂₀₀)-2z₀₁₁*(z₁₀₁+z₁₁₀-z₂₀₀)-z₀₀₂*(z₀₂₀-2z₁₁₀+z₂₀₀)
    
    if denom == 0
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
end # module