module Pebsi

include("Defaults.jl")
include("Geometry.jl")
include("Polynomials.jl")
include("EPMs.jl")
include("Mesh.jl")
include("RectangularMethod.jl")
include("QuadraticIntegration.jl")
# include("Plotting.jl")  # Temporarily disabled due to PyCall issues

end