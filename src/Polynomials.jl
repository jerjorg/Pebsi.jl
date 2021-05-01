module Polynomials

import Base.Iterators: product
import LinearAlgebra: dot,det,norm
import Statistics: mean

@doc """
    sample_simplex(dim,deg)

Get the sample points of a simplex for a polynomial approximation.

# Arguments
- `dim::Integer`: the number of dimensions (2 = triangle, 3 = tetrahedron).
- `deg::Integer`: the degree of the polynomial approximations.

# Returns
- `::AbstractMatrix{<:Real}`: the points on the simplex as columns of an array.

# Examples
```jldoctest
import Pebsi.Polynomials: sample_simplex
dim = 2
deg = 1
sample_simplex(dim,deg)
# output
3×3 Array{Float64,2}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```
"""
function sample_simplex(dim::Integer,deg::Integer,
    rtol::Real=sqrt(eps(1.0)),
    atol::Real=0.0)::AbstractMatrix{<:Real}
    reduce(hcat,filter(x->length(x)>0, 
        [isapprox(sum(p),1,rtol=rtol,atol=atol) ? collect(p) : [] 
        for p=collect(product([0:1/deg:1 for i=0:dim]...))]))
end

@doc """
    bernstein_basis(bpt,dim,deg)

Evaluate the Bernstein polynomials at a point of a given degree and for a given dimension.

# Arguments
- `bpt::AbstractArray`: a point in Barycentric coordinates
- `dim::Integer`: the number of dimensions.
- `deg::Integer`: the degree of the polynomials

# Returns
- `::AbstractArray`: the Bernstein polynomials of a given degree and dimension
    evaluated at the given point in Barycentric coordinates.

# Examples
```jldoctest
import Pebsi.Polynomials: bernstein_basis
import PyCall: pyimport
sympy=pyimport("sympy")
import SymPy: symbols
s,t,u=symbols("s,t,u")
bernstein_basis([s,t,u],2,2)
# output
6-element Array{SymPy.Sym,1}:
 1.0*s^2
 2.0⋅s⋅t
 1.0*t^2
 2.0⋅s⋅u
 2.0⋅t⋅u
 1.0*u^2
```
"""
function bernstein_basis(bpt::AbstractArray,dim::Integer,
    deg::Integer)::AbstractArray
    indices = [p for p=collect(product([0:deg for i=0:dim]...)) if 
        sum(p) == deg]
    [factorial(deg)/prod(factorial.(index))*prod(bpt.^index) for index=indices]
end

"""
    bernstein_basis(bpts,dim,deg)

Evaluate the Bernstein basis functions at each point in an array (points are columns).
"""
function bernstein_basis(bpts::AbstractMatrix{<:Real},dim::Integer,
    deg::Integer)::AbstractMatrix{<:Real}
    mapslices(x->bernstein_basis(x,dim,deg),bpts,dims=1)
end


@doc """
    barytocart(barypt,simplex)

Convert a point from barycentric to Cartesian coordinates.

# Arguments
- `barypt::AbstractVector{<:Real}`: a point in Barycentric coordinates.
- `simplex::AbstractMatrix{<:Real}`: the vertices of a simplex as columns of
    an array.

# Returns
- `::AbstractVector{<:Real}`: the point in Cartesian coordinates.

# Examples
```jldoctest
import Pebsi.Polynomials: barytocart
barypt = [0,0,1]
simplex = [0.0 0.5 0.5; 1.0 1.0 0.0]
barytocart(barypt,simplex)
# output
2-element Array{Float64,1}:
 0.5
 0.0
```
"""
function barytocart(barypt::AbstractVector{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractVector{<:Real}
    [sum(reduce(hcat,[simplex[:,i]*barypt[i] for i=1:length(barypt)]),dims=2)...]
end

@doc """
    barytocart(barypts,simplex)

Convert points as columns on an array from barycentric to Cartesian coordinates.
"""
function barytocart(barypts::AbstractMatrix{<:Real},
    simplex::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    mapslices(x->barytocart(x,simplex),barypts,dims=1)
end

@doc """
    carttobary(pt,simplex)

Transform a point from Cartesian to barycentric coordinates.

# Arguments
- `pt::AbstractVector{<:Real}`: the point in Cartesian coordinates
- `simplex::AbstractMatrix{<:Real}`: the corners of the simplex as columns of an array.
"""
function carttobary(pt::AbstractVector{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractVector{<:Real}

    inv(vcat(simplex,ones(Int,(1,size(simplex,2)))))*vcat(pt,1)
end

@doc """
    carttobary(pt,simplex)

Transform an array of points from Cartesian to barycentric coordinates.

# Arguments
- `pts::AbstractMatrix{<:Real}`: the points in Cartesian coordinates as columns of an array.
"""
function carttobary(pts::AbstractMatrix{<:Real},
        simplex::AbstractMatrix{<:Real})::AbstractMatrix{<:Real}
    mapslices(x->carttobary(x,simplex),pts,dims=1)
end

@doc """
    getpoly_coeffs(values,simplex_bpts,dim,deg)

Calculate the coefficients of a polynomial interpolation over a simplex.

# Arguments
- `values::AbstractVector{<:Real}`: the value of the approximated function at
    samples within the simplex.
- `simplex_bpts::AbstractMatrix{<:Real}`: the sample points in the simplex in 
    barycentric coordinates as columns of an array.
- `dim::Integer`: the number of dimensions.
- `deg::Integer`: the degree of the polynomial.

# Returns
- `::AbstractVector{<:Real}`: the coefficients of the polynomial approximation.

# Examples
```jldoctest
import Pebsi.Polynomials: getpoly_coeffs
simplex_bpts = [1.0 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
values = [0.4, 0.5, 0.3, -0.4, -0.1, -0.2]
dim = 2
deg = 2
getpoly_coeffs(values,simplex_bpts,dim,deg)
# output
6-element Array{Float64,1}:
  0.4
  0.65
  0.3
 -0.9
 -0.25
 -0.2
```
"""
function getpoly_coeffs(values::AbstractVector{<:Real},
    simplex_bpts::AbstractMatrix{<:Real},dim::Integer,
    deg::Integer)::AbstractVector{<:Real}
    inv(mapslices(x->bernstein_basis(x,dim,deg),simplex_bpts,dims=1)')*values
end

@doc """
    eval_poly(barypt,coeffs,dim,deg)

Evaluate a polynomial at a point.

# Arguments
- `barypt::AbstractVector{<:Real}`: a point in Barycentric coordinates.
- `coeffs::AbstractVector{<:Real}`: the coefficients of the polynomial approximation
- `dim::Integer`: the number of dimensions.
- `deg::Integer`: the degree of the polynomial.

# Returns
` `::Any`: the value of the polynomial approximation an `barypt`.

# Examples
```jldoctest
barypt = [1,0,0]
coeffs = [0.4, 0.65, 0.3, -0.9, -0.25, -0.2]
dim = 2
deg = 2
eval_poly(barypt,coeffs,dim,deg)
# output
0.4
```
"""
function eval_poly(barypt::AbstractArray,
    coeffs::AbstractVector{<:Real},dim::Integer,deg::Integer)::Any
    dot(coeffs,bernstein_basis(barypt,dim,deg))
end

function eval_poly(barypt::AbstractArray,
    coeffs::AbstractArray,dim::Integer,deg::Integer)::Any
    dot(coeffs,bernstein_basis(barypt,dim,deg))
end

@doc """
    eval_poly(barypts,coeffs,dim,deg)

Evaluate a polynomial for each point in an array (points are columns).

# Examples
```jldoctest
simplex_bpts = [1.0 0.5 0.0 0.5 0.0 0.0; 0.0 0.5 1.0 0.0 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0]
coeffs = [0.4, 0.5, 0.4, -0.2, -0.1, -0.3]
dim = 2
deg = 2
eval_poly(simplex_bpts,coeffs,dim,deg)
# output
6-element Array{Float64,1}:
  0.4
  0.44999999999999996
  0.4
 -0.075
 -0.024999999999999994
 -0.3
```
"""
function eval_poly(barypts::AbstractMatrix{<:Real},
    coeffs::AbstractVector{<:Real},dim::Integer,
    deg::Integer)::AbstractVector{<:Real}
    mapslices(x->eval_poly(x,coeffs,dim,deg),barypts,dims=1)[:]
end


@doc """
    getbez_pts₋wts(bezpts,p₀,p₂)

Calculate the Bezier points and weights of a level curve of a Quadratic surface passing through two points.

# Arguments
- `bezpts::AbstractVector{<:Real}`: the Bezier points of the quadratic surface.
- `p₀::AbstractVector{<:Real}`: a point a level curve of the quadratic surface passes
    through. The level curve is taken at an isovalue of zero.
- `p₂::AbstractVector{<:Real}`: a point a level curve of the quadratic surface passes
    through. The level curve is taken at an isovalue of zero.

# Returns
- `::::Array{Array{<:Real,N} where N,1}`: the Bezier points and weights in a 1D array.

# Examples
```jldoctest
import Pebsi.getbez_pts₋wts(bezpts,p₀,p₁)
bezpts = [-1.0 0.0 1.0 -0.5 0.5 0.0; 0.0 0.0 0.0 0.5 0.5 1.0; 0.0 1.0 0.0 1.0 -1.0 0.0]
p₀ = [1,0]
p₂ = [0,1]
getbez_pts₋wts(bezpts,p₀,p₂)
# output 
2-element Array{Array{Float64,N} where N,1}:
 [1.0 0.0 0.0; 0.0 0.3333333333333333 1.0]
 [1.0, 1.6770509831248424, 1.0]
```
"""
function getbez_pts₋wts(bezpts::AbstractMatrix{<:Real},
        p₀::AbstractVector{<:Real},
        p₂::AbstractVector{<:Real}; atol::Real=1e-12)

    triangle = bezpts[1:2,[1,3,6]]
    coeffs = bezpts[3,:]
    (z₂₀₀,z₁₁₀,z₀₂₀,z₁₀₁,z₀₁₁,z₀₀₂) = coeffs
    (s₀,t₀,u₀) = carttobary(p₀,triangle)
    (s₂,t₂,u₂) = carttobary(p₂,triangle)

    A₀ = z₂₀₀*s₀ + z₁₁₀*t₀ + z₁₀₁*u₀
    A₂ = z₂₀₀*s₂ + z₁₁₀*t₂ + z₁₀₁*u₂
    B₀ = z₁₁₀*s₀ + z₀₂₀*t₀ + z₀₁₁*u₀
    B₂ = z₁₁₀*s₂ + z₀₂₀*t₂ + z₀₁₁*u₂
    C₀ = z₁₀₁*s₀ + z₀₁₁*t₀ + z₀₀₂*u₀
    C₂ = z₁₀₁*s₂ + z₀₁₁*t₂ + z₀₀₂*u₂

    a₀ = (s₀ + 2t₀ + 2u₀)*A₀ - t₀*B₀ - u₀*C₀
    a₂ = (s₂ + 2t₂ + 2u₂)*A₂ - t₂*B₂ - u₂*C₂
    b₀ = -s₀*A₀ + (2s₀+t₀+2u₀)*B₀ - u₀*C₀
    b₂ = -s₂*A₂ + (2s₂+t₂+2u₂)*B₂ - u₂*C₂
    c₀ = -s₀*A₀ - t₀*B₀ + (2s₀+2t₀+u₀)*C₀
    c₂ = -s₂*A₂ - t₂*B₂ + (2s₂+2t₂+u₂)*C₂

    cₓb = c₀*b₂-c₂*b₀
    aₓc = a₀*c₂-a₂*c₀
    bₓa = b₀*a₂-b₂*a₀
    d = cₓb + aₓc + bₓa
    p₁ = [cₓb,aₓc,bₓa]/d

    (w₀,w₂)=(1,1)
    h₀₀₂ = eval_poly(p₁,coeffs,2,2)
    h₁₁₀ = 2*eval_poly(carttobary((p₀+p₂)/2,triangle),coeffs,2,2)
    cstype = conicsection(bezpts[end,:],atol=atol)    

    # The weight is negative for a straight line.
    if ((isapprox(d,0,atol=atol) && 
        any(cstype .== ["line","rectangular hyperbola","parallel lines"])) ||
        isapprox(h₀₀₂,0,atol=atol))
        bezwtsᵣ = [w₀,0,w₂]
        bezptsᵣ = [p₀ (p₀+p₂)/2 p₂]
    else
        w₁ = √(-h₁₁₀/2h₀₀₂)
        bezwtsᵣ = [w₀,w₁,w₂]
        bezptsᵣ = [p₀ barytocart(p₁,triangle) p₂]
    end
    [bezptsᵣ,bezwtsᵣ]
end

@doc """
    eval_bezcurve(t,bezpts,bezwts)

Evaluate a rational Bezier curve at a point.

# Arguments
- `t::Real`: the parametric variable.
- `bezpts::AbstractMatrix{<:Real}`: the Bezier points in Cartesian coordinates
    as columes of an array.
- `bezwts::AbstractVector{<:Real}`: the weights of the Bezier points.

# Returns
- `::AbstractVector{<:Real}`: a point along the Bezier curve.

# Examples
```jldoctest
import Pebsi.Polynomials: eval_bezcurve
t = 0.5
bezpts = [0.0 0.0 1.0; 1.0 1/3 0.0]
bezwts = [1.0, 1.5, 1.0]
eval_bezcurve(t,bezpts,bezwts)
# output
2-element Array{Float64,1}:
 0.2
 0.4
```
"""
function eval_bezcurve(t::Real,bezpts::AbstractMatrix{<:Real},
        bezwts::AbstractVector{<:Real})::AbstractVector{<:Real}
    (sum((bernstein_basis([1-t,t],1,2).*bezwts)' .* 
        bezpts,dims=2)/dot(bezwts,bernstein_basis([1-t,t],1,2)))[:]
end

@doc """
    eval_bezcurve(t,bezpts,bezwts)

Evaluate a rational Bezier curve at each point in an array.
"""
function eval_bezcurve(t::AbstractVector{<:Real},
    bezpts::AbstractMatrix{<:Real},
    bezwts::AbstractVector{<:Real})::AbstractMatrix{<:Real}
    reduce(hcat,map(x->eval_bezcurve(x,bezpts,bezwts),t))
end

@doc """
    conicsection(coeffs;atol=1e-12)

Classify the conic section of a level curve of a quadratic surface.

# Arguments
- `coeffs::AbstractVector{<:Real}`: the coefficients of the quadratic polynomial.
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
function conicsection(coeffs::AbstractVector{<:Real};
    atol::Real=1e-12)::String
    (z₀₀₂, z₁₀₁, z₂₀₀, z₀₁₁, z₁₁₀, z₀₂₀)=coeffs
    a = z₀₀₂ - 2z₁₀₁ + z₂₀₀
    b = 2z₀₀₂ - 2z₀₁₁ - 2z₁₀₁ + 2z₁₁₀
    c = z₀₀₂ - 2z₀₁₁ + z₀₂₀
    d = b^2 - 4*a*c
    m = -8*(-2*z₀₁₁*z₁₀₁*z₁₁₀+z₀₀₂*z₁₁₀^2+z₀₁₁^2*z₂₀₀+z₀₂₀*(z₁₀₁^2-z₀₀₂*z₂₀₀))

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

end # module