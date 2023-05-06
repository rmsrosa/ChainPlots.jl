using Documenter
using ChainPlots

makedocs(
    sitename = "ChainPlots.jl",
    pages = [
        "Overview" => "index.md"
        "Examples" => [
            "chain_examples.md",
            "metagraphs_examples.md"
        ]
        "API" => "api.md"
    ],
    authors = "Ricardo Rosa",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/rmsrosa/ChainPlots.jl",
        edit_link = "main",
    ),
    modules = [ChainPlots],
)

if get(ENV, "CI", nothing) == "true" || get(ENV, "GITHUB_TOKEN", nothing) !== nothing
    deploydocs(
        repo = "github.com/rmsrosa/ChainPlots.jl.git",
        devbranch = "main",
        forcepush = true,
    )
end

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#=deploydocs(
    repo = "<repository url>"
)=#
