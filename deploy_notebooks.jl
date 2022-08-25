using Literate

for fl in readdir()
    if splitext(fl)[end]!=".jl" || splitpath(@__FILE__)[end]==fl
        continue
    end
    println("File: $fl")
    Literate.notebook(fl, "./", credit=false, execute=false, mdstrings=true)
end
