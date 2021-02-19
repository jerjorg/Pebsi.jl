module Polynomials

import Base.Iterators: product
import LinearAlgebra: dot

"""
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
function sample_simplex(dim::Integer,deg::Integer)::AbstractArray{<:Real,2}
    reduce(hcat,filter(x->length(x)>0, [sum(p)==1 ? collect(p) : []
        for p=collect(product([0:1/deg:1 for i=0:dim]...))]))
end

"""
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
"""
function bernstein_basis(bpt::AbstractArray,dim::Integer,
    deg::Integer)::AbstractArray
    indices = filter(x->length(x)>0, [[sum(p)==deg ? p : []
        for p=collect(product([0:deg for i=0:dim]...))]...])
    [factorial(deg)/prod(factorial.(index))*prod(bpt.^index) for index=indices]
    end

    """
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

"""
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
"""
function getpoly_coeffs(values::AbstractArray{<:Real,1},
    simplex_bpts::AbstractArray{<:Real,2},dim::Integer,
    deg::Integer)::AbstractArray{<:Real,1}
    inv(mapslices(x->bernstein_basis(x,dim,deg),simplex_bpts,dims=1)')*values
end

"""
    eval_poly(barypt,coeffs,dim,deg)

Evaluate a polynomial at a point.

# Arguments
- `barypt::AbstractArray{<:Real,1}`: a point in Barycentric coordinates.
- `coeffs::AbstractArray{<:Real,1}`: the coefficients of the polynomial approximation
- `dim::Integer`: the number of dimensions.
- `deg::Integer`: the degree of the polynomial.

# Returns
` `::Real`: the value of the polynomial approximation an `barypt`.

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
function eval_poly(barypt::AbstractArray{<:Real,1},
    coeffs::AbstractArray{<:Real,1},dim::Integer,deg::Integer)::Real
    dot(coeffs,bernstein_basis(barypt,dim,deg))
end

end
