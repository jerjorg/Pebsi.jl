# Force a non-interactive matplotlib backend before anything loads PyPlot:
# CI runners are headless, and Pebsi pulls in PyPlot through Plotting.
ENV["MPLBACKEND"] = "agg"

include("aqua.jl")
include("RectangularMethod.jl")
include("Polynomials.jl")
include("Geometry.jl")
include("StrategyEnums.jl")
include("QuadraticIntegration.jl")
include("SimpsonQuadrature.jl")
include("Plotting.jl")
include("ScaleInvariance.jl")
include("ThreeDimensional.jl")
include("KnownLimitations.jl")
include("doctests.jl")
