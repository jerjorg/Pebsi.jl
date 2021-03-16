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
- `::AbstractArray{<:Real,2}`: the points on the simplex as columns of an array.

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
    atol::Real=0.0)::AbstractArray{<:Real,2}
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
function bernstein_basis(bpts::AbstractArray{<:Real,2},dim::Integer,
    deg::Integer)::AbstractArray{<:Real,2}
    mapslices(x->bernstein_basis(x,dim,deg),bpts,dims=1)
end


@doc """
    barytocart(barypt,simplex)

Convert a point from barycentric to Cartesian coordinates.

# Arguments
- `barypt::AbstractArray{<:Real,1}`: a point in Barycentric coordinates.
- `simplex::AbstractArray{<:Real,2}`: the vertices of a simplex as columns of
    an array.

# Returns
- `::AbstractArray{<:Real,1}`: the point in Cartesian coordinates.

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
function barytocart(barypt::AbstractArray{<:Real,1},
        simplex::AbstractArray{<:Real,2})::AbstractArray{<:Real,1}
    [sum(reduce(hcat,[simplex[:,i]*barypt[i] for i=1:length(barypt)]),dims=2)...]
end

@doc """
    barytocart(barypts,simplex)

Convert points as colums on an array from barycentric to Cartesian coordinates.
"""
function barytocart(barypts::AbstractArray{<:Real,2},
    simplex::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    mapslices(x->barytocart(x,simplex),barypts,dims=1)
end

@doc """
    carttobary(pt,simplex)

Transform a point from Cartesian to barycentric coordinates.

# Arguments
- `pt::AbstractArray{<:Real,1}`: the point in Cartesian coordinates
- `simplex::AbstractArray{<:Real,2}`: the corners of the simplex as columns of an array.
"""
function carttobary(pt::AbstractArray{<:Real,1},
        simplex::AbstractArray{<:Real,2})::AbstractArray{<:Real,1}

    inv(vcat(simplex,ones(Int,(1,size(simplex,2)))))*vcat(pt,1)
end

@doc """
    carttobary(pt,simplex)

Transform an array of points from Cartesian to barycentric coordinates.

# Arguments
- `pts::AbstractArray{<:Real,2}`: the points in Cartesian coordinates as columns of an array.
"""
function carttobary(pts::AbstractArray{<:Real,2},
        simplex::AbstractArray{<:Real,2})::AbstractArray{<:Real,2}
    mapslices(x->carttobary(x,simplex),pts,dims=1)
end

@doc """
    getpoly_coeffs(values,simplex_bpts,dim,deg)

Calculate the coefficients of a polynomial interpolation over a simplex.

# Arguments
- `values::AbstractArray{<:Real,1}`: the value of the approximated function at
    samples within the simplex.
- `simplex_bpts::AbstractArray{<:Real,2}`: the sample points in the simplex in 
    barycentric coordinates as columns of an array.
- `dim::Integer`: the number of dimensions.
- `deg::Integer`: the degree of the polynomial.

# Returns
- `::AbstractArray{<:Real,1}`: the coefficients of the polynomial approximation.

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
function getpoly_coeffs(values::AbstractArray{<:Real,1},
    simplex_bpts::AbstractArray{<:Real,2},dim::Integer,
    deg::Integer)::AbstractArray{<:Real,1}
    inv(mapslices(x->bernstein_basis(x,dim,deg),simplex_bpts,dims=1)')*values
end

@doc """
    eval_poly(barypt,coeffs,dim,deg)

Evaluate a polynomial at a point.

# Arguments
- `barypt::AbstractArray{<:Real,1}`: a point in Barycentric coordinates.
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the polynomial approximation
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
    coeffs::AbstractArray{<:Real,1},dim::Integer,deg::Integer)::Any
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
function eval_poly(barypts::AbstractArray{<:Real,2},
    coeffs::AbstractArray{<:Real,1},dim::Integer,
    deg::Integer)::AbstractArray{<:Real,1}
    mapslices(x->eval_poly(x,coeffs,dim,deg),barypts,dims=1)[:]
end


@doc """
    getbez_pts₋wts(bezpts,p₀,p₂)

Calculate the Bezier points and weights of a level curve of a Quadratic surface passing through two points.

# Arguments
- `bezpts::AbstractArray{<:Real,1}`: the Bezier points of the quadratic surface.
- `p₀::AbstractArray{<:Real,1}`: a point a level curve of the quadratic surface passes
    through. The level curve is taken at an isovalue of zero.
- `p₂::AbstractArray{<:Real,1}`: a point a level curve of the quadratic surface passes
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
function getbez_pts₋wts(bezpts::AbstractArray{<:Real,2},
        p₀::AbstractArray{<:Real,1},
        p₂::AbstractArray{<:Real,1})#::Array{Array{<:Real,N} where N,1}

    triangle = bezpts[1:2,[1,3,6]]
    coeffs = bezpts[3,:]
    (z₂₀₀,z₁₀₁,z₀₀₂,z₁₁₀,z₀₁₁,z₀₂₀) = coeffs
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
    w₁ = √(-h₁₁₀/2h₀₀₂)

    bezwtsᵣ = [w₀,w₁,w₂]
    bezptsᵣ = [p₀ barytocart(p₁,triangle) p₂]
    [bezptsᵣ,bezwtsᵣ]
end


@doc """
    eval_bezcurve(t,bezpts,bezwts)

Evaluate a rational Bezier curve at a point.

# Arguments
- `t::Real`: the parametric variable.
- `bezpts::AbstractArray{<:Real,2}`: the Bezier points in Cartesian coordinates
    as columes of an array.
- `bezwts::AbstractArray{<:Real,1}`: the weights of the Bezier points.

# Returns
- `::AbstractArray{<:Real,1}`: a point along the Bezier curve.

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
function eval_bezcurve(t::Real,bezpts::AbstractArray{<:Real,2},
        bezwts::AbstractArray{<:Real,1})::AbstractArray{<:Real,1}
    (sum((bernstein_basis([1-t,t],1,2).*bezwts)' .* 
        bezpts,dims=2)/dot(bezwts,bernstein_basis([1-t,t],1,2)))[:]
end

@doc """
    eval_bezcurve(t,bezpts,bezwts)

Evaluate a rational Bezier curve at each point in an array.
"""
function eval_bezcurve(t::AbstractArray{<:Real,1},
    bezpts::AbstractArray{<:Real,2},
    bezwts::AbstractArray{<:Real,1})::AbstractArray{<:Real,2}
    reduce(hcat,map(x->eval_bezcurve(x,bezpts,bezwts),t))
end

end # module