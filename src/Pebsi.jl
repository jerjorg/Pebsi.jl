module Pebsi

include("EPMs.jl")
include("Plotting.jl")
include("RectangularMethod.jl")
include("Polynomials.jl")
include("QuadraticIntegration.jl")
import .EPMs: eval_epm
export eval_epm

end
