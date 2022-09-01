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
Pkg.add(url="https://github.com/EnzymeAD/Enzyme.jl.git", rev="main")
```
_:warning: Note that the `main` branch of Enzyme.jl should be used._

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


## Self-contained adjoint-based GD inversion scripts
The scripts used to produce the adjoint-based gradient descent point-wise inversion for the power-law prefactor in a free-surface channel flow are accessible in the [scripts](scripts) folder:
- GPU version: [free_surface_flow_enzyme_2D_cuda.jl](scripts/free_surface_flow_enzyme_2D_cuda.jl)
- CPU version: [free_surface_flow_enzyme_2D.jl](scripts/free_surface_flow_enzyme_2D.jl)


## Resources

#### The pseudo-transient method
GMD paper: https://doi.org/10.5194/gmd-15-5757-2022
- Stokes flow examples: https://github.com/PTsolvers/PseudoTransientStokes.jl
- Diffusion solver examples: https://github.com/PTsolvers/PseudoTransientDiffusion.jl

#### Julia packages
- [ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)
- [ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)
- [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- [Enzyme.jl](https://github.com/EnzymeAD/Enzyme.jl)

#### Courses and resources
- ETHZ course on solving PDEs with GPUs: https://pde-on-gpu.vaw.ethz.ch
- More [here](https://pde-on-gpu.vaw.ethz.ch/extras/#extra_material)

#### Misc
- Frontier GPU multi-physics solvers: https://ptsolvers.github.io/GPU4GEO/
