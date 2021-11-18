module Pebsi

include("Defaults.jl")
include("Geometry.jl")
include("Polynomials.jl")
include("EPMs.jl")
include("Mesh.jl")
include("RectangularMethod.jl")
include("QuadraticIntegration.jl")
include("Plotting.jl")

# Geometry
export order_vertices!, sample_simplex, barytocart, carttobary, simplex_size, 
    insimplex, lineseg₋pt_dist, ptface_mindist, affine_trans, mapto_xyplane,
    point_in_polygon
    
# Polynomials
export bernstein_basis, getpoly_coeffs, eval_poly, getbez_pts₋wts, 
    eval_bezcurve, conicsection, eval_1Dquad_basis, get_1Dquad_coeffs,
    evalpoly1D, solve_quadratic

# EPMs
export epm_names, epm_names2D, eval_epm, epm₋model2D, epm₋model, free, free_fl,
    free_be, free2D, free_fl2D, free_be2D, epms, epms2D, m11, m12, m13, m21, m22, 
    m23, m31, m32, m33, m41, m42, m43, m51, m52, m53, mf, Ag_epm, Al_epm, Au_epm, Cs_epm, Cu_epm, In_epm,
    K_epm, Li_epm, Na_epm, Pb_epm, Rb_epm, Sn_epm, Zn_epm, free_epm

# Mesh
export get_neighbors, choose_neighbors, choose_neighbors3D, ibz_init₋mesh,
    get_sym₋unique!, notbox_simplices, get_cvpts, get_extmesh, trimesh

# RectangularMethod
export sample_unitcell, rectangular_method, symreduce_grid, convert_mixedradix,
    kpoint_index, calculate_orbits

# QuadraticIntegration
export bandstructure, init_bandstructure, quadval_vertex, corner_indices, 
    edge_indices, simplex_intersects, saddlepoint, split_bezsurf₁, 
    split_bezsurf, analytic_area, analytic_volume, sub₋coeffs,
    two₋intersects_area₋volume, quad_area₋volume, get_intercoeffs, calc_fl,
    calc_flbe!, refine_mesh!, get_tolerances, quadratic_method!, truebe,
    bezcurve_intersects, getdomain, analytic_area1D, simpson, simpson2D, 
    linept_dist, tetface_areas, simpson3D, quadslice_tanpt

# Plotting
export meshplot, contourplot, bezplot, bezcurve_plot, polygonplot, 
    plot_bandstructure

end