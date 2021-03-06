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
function edge_intersects(bezpts::AbstractArray{<:Real,2})::AbstractArray{<:Real,1}
    
    # Cases where the curve is above zero, below zero, or at zero.
    coeffs = bezpts[end,:]
    if all(coeffs.==0) || all(coeffs .> 0) || all(coeffs .< 0)
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
        x = [(-β-sqrt(β^2-4α*γ))/(2γ),(-β+sqrt(β^2-4α*γ))/(2γ)]
    end
    
    # Only keep non-complex intersections between [0,1).
    real.(filter(y -> imag(y)==0 && real(y)>=0 && real(y)<1 ,x)) |> sort
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
    intersects
end

end # module