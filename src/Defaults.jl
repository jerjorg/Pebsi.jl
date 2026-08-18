module Defaults

# Strategy selectors. These were bare integers compared against literals at 21
# sites, with the meaning of each number recorded only in a comment here. The
# numeric values are kept so the mapping to the published work is unchanged.
@enum FermiLevelMethod begin
    fl_bisection = 1
    fl_chandrupatla = 2
end

@enum RefineMethod begin
    refine_most_error = 1              # the tiles with the most error
    refine_above_allowed = 2           # tiles with too much error for their size
    refine_fraction_above_allowed = 3  # a fraction of the tiles with too much error
    refine_fermiarea_above_allowed = 4 # by Fermi area error, scaled to band energy
    refine_fermiarea_largest = 5       # largest Fermi area errors, no allowance
    refine_partially_occupied = 6      # a fraction of the partially occupied tiles
    refine_largest_fraction = 7        # a fraction of the largest errors, capped
end

@enum SampleMethod begin
    sample_center = 1          # one point at the centre of the triangle
    sample_edge_midpoints = 2  # points at the midpoints of every edge
    sample_adaptive = 3        # edges when the error is large, otherwise centre
end

@enum NeighborMethod begin
    neighbors_closest = 1      # the neighbours nearest the triangle
    neighbors_surrounding = 2  # near the triangle and spread around it
    neighbors_inside = 3       # from a uniform grid within the triangle
end

@enum StopCriterion begin
    stop_total_error = 1     # summed band energy error below the target
    stop_energy_change = 2   # change in band energy below the target
    stop_interval = 3        # Fermi area interval below the target
    stop_kpoint_target = 4   # k-point count close enough to the target
end

# Default values, tolerances, and hyperparameter constants
const def_init_msize = 3 # The initial size of the mesh
const def_num_near_neigh = 2 # the number of nearest neighbors included in neighbor calculations
const def_fermiarea_eps = 1e-10 # The convergence tolerance for the Fermi area
const def_target_accuracy = 1e-4 # The target accuracy for the band energy
const def_fermilevel_method = fl_chandrupatla # Chandrupatla's root finding algorithm
const def_refine_method = refine_largest_fraction # Split a fraction of triangles with more than allowed error
const def_frac_refined = 0.2 # The fraction of triangles refined
const def_sample_method = sample_edge_midpoints # Add sample points at midpoints of edges
const def_neighbor_method = neighbors_surrounding # Select neighbors close and surrounding the triangle
const def_uniform = false # Do adaptive refinement by default
const def_rtol = 1e-9 # Relative tolerance for floating point comparisons
const def_atol = 1e-9 # absolute tolerance for floating point comparisons
# Polynomial coefficients count as zero below this fraction of the largest
# coefficient in the same polynomial, rather than below a fixed number.
#
# These coefficients are eigenvalues measured from the Fermi level, so they are
# smallest exactly where the Fermi surface crosses a triangle - the triangles
# that matter most. Comparing them against an absolute 1e-9 discards the level
# set of any polynomial whose coefficients are small: scaling the coefficients
# of a test patch by 1e-10, which does not move the zero contour or change the
# true area at all, made the computed area 3.5 times too large, because no
# intersections were found and the whole triangle was reported as occupied.
const def_coeff_rtol = 1e-9

# A coefficient is roundoff rather than merely small when it falls below this
# many multiples of the working precision, measured against the reference
# coefficient scale carried down from the patch the calculation started on.
#
# Two references are needed because a tolerance relative only to the coefficients
# present cannot tell a small number from a meaningless one - dividing noise by
# itself makes the noise look significant. Subdividing a patch whose values are of
# order one leaves sub-patches whose coefficients are cancellation residue, and
# judged on their own terms those become a genuine quadratic, so a region that is
# not there gets integrated.
#
# The floor is expressed in multiples of `eps` rather than as a fixed number
# because it is a property of the arithmetic, not of the problem. The same
# coefficient that is unrecoverable in Float64 may be perfectly meaningful in
# higher precision, and taking eps from the coefficient type means the floor
# drops accordingly - which is what makes computing in BigFloat worth doing here
# rather than merely slower.
const def_coeff_noise_eps = 100

# How close a root may come to the ends of its edge before it counts as sitting
# on the corner. This is a question about a curve parameter, which runs from 0 to
# 1 whatever the geometry or the coefficients, so it is genuinely absolute - and
# it is a different question from whether a coefficient is zero, which is why it
# no longer shares that tolerance.
const def_root_boundary_atol = 1e-9
const def_fatten = 2.0 # A parameter for scaling the interval coefficients
const max_refine_steps = 100 # The maximum number of refinement iterations
const def_num_neighbors2D = 16 # The desired number of neighbors in 2D interval coefficient calculation
const def_num_neighbors3D = 60 # The desired number of neighbors in 3D interval coefficient calculation
const def_neighbors_per_bin2D = 2 # The number of neighbors per bin (angle range) in 2D
const def_neighbors_per_bin3D = 3 # The number of neighbors per bin (angle range) in 3D
const def_mesh_scale = 100 # Determines the size of a box that surrounds the triangle or mesh
const def_taylor_exp_tol = 1e-2 # Tolerance for analytic areas and volumes when weight close to one
const def_fl_max_iters = 50 # The maximum number root-finding iterations for Fermi level calculation
const def_chandrupatla_tol = 1e-2 # Tolerance for Chandrupatla's method when t is close to zero or 1
const def_min_split = 10 # The minimum number of simplices split per refinement iteration
const def_allowed_err_ratio = 5 # Cutoff between adding one or three/six sample points in refinement
# Two candidate neighbours count as the same distance from a simplex when their
# distances agree to within this fraction of that simplex's shortest edge.
#
# A symmetric mesh puts many candidates at the same distance, and which of them
# get kept then depends on how the sort orders equal values. Comparing at full
# precision makes that depend on the last bits of the geometry, which a different
# BLAS will move - the mechanism behind calc_fl differing between Julia versions.
#
# The tolerance is relative to the local edge length rather than absolute because
# adaptive refinement leaves triangles differing in size by orders of magnitude,
# so no single absolute distance is meaningful across the mesh.
const def_neighbor_dist_rtol = 1e-9
const def_max_neighbor_tol = 1.01 # Tolerance for selecting neighbors near the triangle
const def_inside_neighbors_divs = 5 # The number of points for a uniform grid over a triangle for inside neighbors
const def_bez_weight_tol = 1e-12 # Smaller tolerance for classifying conic sections
# A patch this small that `split_bezsurf` could not reduce to two curve
# intersections is integrated by its sign instead of exactly.
#
# The box-padded Delaunay in split_bezsurf1 sometimes cannot subdivide a triangle
# even though it is well above def_min_simplex_size - every candidate sub-triangle
# has a corner on the padding box - and the result comes back with three
# intersections, which two_intersects_area_volume has no formula for. Such a patch
# contributes at most its own size to the area, and at most its size times its
# largest coefficient to the volume, so approximating it costs no more than that.
# Refusing to integrate it costs the entire calculation. Above this size the
# refusal stands, since a large patch that will not subdivide means something has
# gone wrong rather than merely become small.
const def_degenerate_simplex_size = 1e-6
# A simplex from a triangulation counts as degenerate when its size is below
# this fraction of the largest simplex in the same triangulation.
#
# This test used to be absolute - `isapprox(size, 0, atol=def_atol)` with def_atol
# at 1e-9 - which silently made the whole splitting algorithm scale dependent.
# Subdividing a patch below about 1e-8 in area produced perfectly good children,
# every one of them smaller than 1e-9, so all of them were discarded as
# "zero volume" and the caller was told the patch could not be subdivided. The
# geometry was never the problem; the yardstick was.
const def_simplex_size_rtol = 1e-9
const def_min_simplex_size = 1e-12 # The smallest triangle that can be split
const def_rational_bezpt_dist = 1e6 # The maximum size of a component of a rational Bezier point
const def_weighted = false # Points are not weighted to calculate interval coefficients
const def_constrained = true # Band structure interpolated with constrained least squares
const def_stop_criterion = stop_kpoint_target # The default method used to determine if AMR may stop
const def_target_kpoints = 100 # The default number of k-points for stop_criterion = 4
const def_stop_kpoint_tol = 0.05 # AMR may stop when the k-point count is within this fraction of the target
const def_deriv_step = 1e-4 # The step size for numerical derivatives
const def_num_slices = 100 # The number of slices for integration in 3D
const def_initmesh_kpoint_tol = 0.05 # The IBZ mesh k-point count is within this fraction of the number requested
const def_num_kpoints = 100 # The default number of k-points

# Export all
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end
end # module