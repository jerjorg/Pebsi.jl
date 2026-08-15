# Pebsi.jl

`Pebsi.jl` is a Julia package for electronic band structure integration (PEBSI). Its
primary function is to compute the band energy of empirical pseudopotentials using adaptive
meshes and piece-wise quadratic polynomials. Mesh refinement is driven by the approximation
errors of the quadratic polynomials, so *k*-points are added where they are most beneficial,
that is, where there is the most error. The goal is to compute the band energy as accurately
as possible with as few *k*-points as possible, and to approximate the error in the band
energy for a given level of refinement.

`Pebsi.jl` also contains methods in 2D and 3D for:

- Creating empirical pseudopotential models on which to test Brillouin zone integration
- Generating Monkhorst-Pack or generalized regular *k*-point grids
- Computing the symmetrically-unique *k*-points of regular grids
- Computing the Fermi level and band energy with the rectangular method
- Creating *k*-point meshes over the irreducible Brillouin zone
- Interpolating the band structure with piece-wise quadratic interval polynomials
- Adaptive refinement of *k*-point meshes driven by band energy errors
- Computing the Fermi level and band energy with quadratic polynomials

Irreducible Brillouin zones are constructed with
[SymmetryReduceBZ.jl](https://github.com/jerjorg/SymmetryReduceBZ.jl).

## Installation

`Pebsi.jl` is not in the Julia General registry and is installed directly from this
repository:

```julia
using Pkg
Pkg.add(url="https://github.com/jerjorg/Pebsi.jl")
```

## Status

This is research code written alongside work on adaptive Brillouin zone integration. It is
functional but not under active development.
