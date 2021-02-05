using Pebsi
using Documenter

makedocs(;
    modules=[Pebsi],
    authors="Jeremy Jorgensen",
    repo="https://github.com/jerjorg/Pebsi.jl/blob/{commit}{path}#L{line}",
    sitename="Pebsi.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://jerjorg.github.io/Pebsi.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jerjorg/Pebsi.jl",
)
