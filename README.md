# Ada Lovelace 2022 GPU
GPU-computing in geodynamics workshop material

Program: https://meetings.copernicus.org/2022AdaLovelaceWorkshop/programme/programme.html

## Automatic notebook generation

The presentation slides and the demo notebook are self-contained in a Jupyter notebook [alw22-gd-gpu.ipynb](alw22-gd-gpu.ipynb) that can be auto-generated using literate programming by deploying the [alw22-gd-gpu.jl](alw22-gd-gpu.jl) script.

To reproduce:
1. Clone this git repo
2. Open Julia and resolve/instantiate the project
```julia-repl
using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()
```
3. Run the deploy script
```julia-repl
julia> using Literate

julia> include("deploy_notebooks.jl")
```
4. Then using IJulia, you can launch the notebook and get it displayed in your web browser:
```julia-repl
julia> using IJulia

julia> notebook(dir="./")
```
_To view the notebook as slide, you need to install the [RISE](https://rise.readthedocs.io/en/stable/installation.html) plugin_

## Resources

#### Packages
- ...

#### Courses and resources
- ETHZ course on solving PDEs with GPUs: https://pde-on-gpu.vaw.ethz.ch
- More [here](https://pde-on-gpu.vaw.ethz.ch/extras/#extra_material)

#### Misc
- Frontier GPU multi-physics solvers: https://ptsolvers.github.io/GPU4GEO/
