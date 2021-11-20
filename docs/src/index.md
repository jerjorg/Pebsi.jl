# Outline

`Pebsi.jl` is a Julia package for electronic band structure integration (PEBSI). Its primary function is to compute the band energy of empirical pseudopotentials using adaptive meshes and piece-wise quadratic polynomials. Mesh refinement is driven by approximation errors of the quadratic polynomials so that *k*-points are added where they are most beneficial (where there is the most error). The goal is to compute the band energy as accurately as possible with as few *k*-points as possible and approximate the error in the band energy for a given level or refinement.

`PEBSI` also contains methods in 2D and 3D for
- Creating empirical pseudopotential models on which to test Brillouin zone integration
- Generating Monkhorst-Pack or generalized regular *k*-point grids
- Computing the symmetrically-unique *k*-points of regular grids
- Computing the Fermi level and band energy with the rectangular method
- Creating *k*-point meshes over the irreducible Brillouin zone
- Interpolating the band structure with piece-wise quadratic interval polynomials
- Adaptive refinement of *k*-point meshes driven by band energy errors
- Computing the Fermi level and band energy with quadratic polynomials
