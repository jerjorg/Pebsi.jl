module Defaults

# Default values, tolerances, and hyperparameter constants
def_init_msize = 3 # The initial size of the mesh
def_num_neigh = 2 # the number of neighbors included
def_fermiarea_eps = 1e-10 # The convergence param. for the Fermi area
def_target_accuracy = 1e-4 # The target accuracy for the band energy
def_fermilevel_method = 2 # Chandrupatla's root finding algorithm
def_refine_method = 3 # Split a fraction of triangles with more than allowed error
def_frac_refined = 0.1 # The fraction of triangles refined
def_sample_method = 3 # Add one or three sample points depending on mag. of error ratio
def_neighbor_method = 2 # Select neighbors close and surrounding the triangle
def_inside = false # Don't select neighbors from inside the triangle
def_uniform = false # Do adaptive refinement by default
def_rtol = 1e-9 # Relative tolerance for floating point comparisons
def_atol = 1e-9 # absolute tolerance for floating point comparisons
def_fatten = 0.15 # A parameter for scaling the interval coefficients
max_refine_steps = 100 # The maximum number of refinement iterations
def_neighbor_cutoff = 15 # The desired number of neighbors in interval coefficient calculation
def_neighbors_per_bin = 2 # The number of neighbors per bin (angle range)
def_mesh_scale = 100 # Determines the size of square that surrounds the triangle or mesh
def_taylor_exp_tol = 1e-2 # Tolerance for analytic areas and volumes when weight close to zero
def_fl_max_iters = 50 # The maximum number root-finding iterations for Fermi level calculation
def_chandrupatla_tol = 1e-2 # Tolerance for Chandrupatla's method when t is close to zero or 1
def_min_split_triangles = 10 # The minimum number of triangles split for refinement
def_allowed_err_ratio = 5 # Cutoff between adding one or three sample points in refinement
def_max_neighbor_tol = 1.01 # Tolerance for selecting neighbors near the triangle
def_inside_neighbors_divs = 5 # The number of points for uniform grid over triangle for inside neighbors
def_bez_weight_tol = 1e-12 # Smaller tolerance for classifying conic sections

# Export all
for n in names(@__MODULE__; all=true)
    if Base.isidentifier(n) && n ∉ (Symbol(@__MODULE__), :eval, :include)
        @eval export $n
    end
end
end