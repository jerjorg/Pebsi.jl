push!(LOAD_PATH,"../src/")
using Documenter, Pebsi

makedocs(sitename="Pebsi.jl",
         modules = [Pebsi.EPMs, Pebsi.Geometry, Pebsi.Mesh, Pebsi.Plotting, Pebsi.Polynomials, Pebsi.QuadraticIntegration, Pebsi.RectangularMethod],
         authors = "Jeremy Jorgensen",
         doctest = true,
         pages=["index.md", "documentation.md", "contents.md"])