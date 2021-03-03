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

end # module