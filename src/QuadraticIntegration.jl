module QuadraticIntegration

using SymmetryReduceBZ.Utilities
using SymmetryReduceBZ.Utilities: unique_points, shoelace, remove_duplicates, get_uniquefacets
using SymmetryReduceBZ.Symmetry: calc_spacegroup

using ..Polynomials: eval_poly,getpoly_coeffs,getbez_pts_wts,eval_bezcurve,
    conicsection, evalpoly1D, get_1Dquad_coeffs, solve_quadratic, bernstein_basis
using ..EPMs: eval_epm, EPM, EPM2D
using ..Mesh: get_neighbors, notbox_simplices, get_cvpts, ibz_init_mesh, 
    get_extmesh, choose_neighbors, choose_neighbors3D, trimesh, ntripts, ntetpts,
    get_sym_unique!, simplex_cornerpts, ibz_initmesh, ibz_borders,
    bz_translations
using ..Geometry: order_vertices!, simplex_size, insimplex, barytocart,
    carttobary, sample_simplex, lineseg_pt_dist, mapto_xyplane, ptface_mindist
using ..Defaults

using QHull: chull, Chull
using LinearAlgebra: cross, norm, dot, I, diagm, pinv, det
using Statistics: mean
using Base.Iterators: flatten
using PyCall: PyObject, pyimport
using Distributed: pmap
using FastGaussQuadrature: gausslegendre

export BandStructure, init_bandstructure, quadval_vertex, corner_indices, 
    edge_indices, simplex_intersects, saddlepoint, split_bezsurf₁, 
    split_bezsurf, analytic_area, analytic_volume, sub_coeffs,
    two_intersects_area_volume, quad_area_volume, get_intercoeffs, calc_fl,
    calc_flbe!, refine_mesh!, get_tolerances, quadratic_method, truebe, 
    bezcurve_intersects, getdomain, analytic_area1D, simpson, 
    linept_dist, tetface_areas, simpson3D, quadslice_tanpt, containment_percentage,
    stop_refinement!, calc_fabe, quadlin_esterr, length_area1D, area_volume2D,
    volume_hypvol3D, init_exactfit, cubequad_esterr, kpoint_weights


# Implementation, split by concern. These are included rather than made into
# submodules: the surface, area/volume and quadrature routines call each other
# in a cycle, so they cannot be separated without redesign, and keeping one
# module leaves every existing import working.
include("QuadraticIntegration/Surfaces.jl")
include("QuadraticIntegration/Interpolation.jl")
include("QuadraticIntegration/FermiLevel.jl")
include("QuadraticIntegration/Refinement.jl")
include("QuadraticIntegration/Diagnostics.jl")

end # module
