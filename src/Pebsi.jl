module Pebsi

include("Polynomials.jl")
include("EPMs.jl")
include("Geometry.jl")
include("Mesh.jl")
include("Plotting.jl")
include("QuadraticIntegration.jl")
include("RectangularMethod.jl")

import .EPMs: eval_epm
export eval_epm

end
