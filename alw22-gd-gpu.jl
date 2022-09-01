#src # This is needed to make this run as normal Julia file
using Markdown #src

#src ## Talk outline
#src 
#src 1. What we will achieve today
#src - inversion for power-law prefactor in a free-surface channel flow
#src 
#src 2. The (yet invisible) cool stuff
#src - Runs on graphics cards using the Julia language
#src - Fully local and iterative approach (scalability)
#src - Inverse problem automatically retrieved using Automatic differentiation
#src 
#src 3. Why to still bother with GPU computing in 2022
#src - It's around for more than a decade
#src - Shows massive performance gain compared to serial CPU computing
#src - First exascale supercomputer, Frontier, is full of GPUs
#src 
#src 3.1 Performance, what matters
#src - Memory throughput
#src - Non-redundant memory access
#src - 2D diffusion demo on A100
#src 
#src 4. Challenge
#src - But very few software uses it efficiently
#src - Why? It requires to rethink the solving strategy as non-local operations will kill the fun
#src 
#src 5. That's what we address in this workshop
#src - we show that both forward and inverse solvers can efficiently run on GPUs
#src - we demonstrate it by making an inversion of close-to Stokes equations
#src - we develop all this using the Julia language as it solves the "two-language problem"
#src 
#src 6. Outline
#src     1. The accelerated pseudo-transient method - a physics-motivated explanation
#src     2. Application of the PT method to GPU supercomputing
#src     3. Revisiting the adjoint method implementation using automatic differentiation (AD) and the PT method
#src     4. Application: Point-wise inversion for power-law prefactor in a free-surface channel flow
#src     5. Outlook and conclusion
#src     6. Curious? Some useful resources

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# Using graphics cards to solve forward and inverse problems in geodynamics

#### 2022 Ada Lovelace Workshop
#### Hévíz | Hungary | 28 August–2 September 2022

### Ludovic Räss & Ivan Utkin

![eth logo](./figures/logo2.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## What we will achieve today

Inversion for viscosity in a free-surface channel flow

![inversion](./figures/inversion.gif)
"""
#src memo: gifsicle --loop=1 -d 50 inversion.gif > inversion1.gif

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## The (yet invisible) cool stuff

The code used to produce the inversion
- Runs on graphics cards using the Julia language
- Uses a fully local and iterative approach (scalability)
- Retrieves automatically the "inverse" (adjoint) variables using automatic differentiation (AD)
- (Features 340 lines of code - 3 solvers + GD)
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Too good to be true? Hold on 🙂 ...
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Why to still bother with GPU computing in 2022

- It's around for more than a decade
- Shows massive performance gain compared to serial CPU computing
- First exascale supercomputer, Frontier, is full of GPUs

![Frontier](./figures/frontier.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Performance that matters

![cpu_gpu_evo](./figures/cpu_gpu_evo.png)

Taking a look at a recent GPU and CPU:
- Nvidia Tesla A100 GPU
- AMD EPYC "Rome" 7282 (16 cores) CPU

| Device         | TFLOP/s (FP64) | Memory BW TB/s | Imbalance (FP64)     |
| :------------: | :------------: | :------------: | :------------------: |
| Tesla A100     | 9.7            | 1.55           | 9.7 / 1.55  × 8 = 50 |
| AMD EPYC 7282  | 0.7            | 0.085          | 0.7 / 0.085 × 8 = 66 |
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
**Meaning:** we can do about 50 floating point operations per number accessed from main memory.
Floating point operations are "for free" when we work in memory-bounded regimes.

👉 Requires to re-think the numerical implementation and solution strategies
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
Unfortunately, the cost of evaluating a first derivative $∂A / ∂x$ using finite-differences

```julia
q[ix] = -D*(A[ix+1]-A[ix])/dx
```

consists of:
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
- 1 reads + 1 write => $2 × 8$ = **16 Bytes transferred**
- 1 (fused) addition and division => **1 floating point operations**

👉 assuming $D$, $∂x$ are scalars, $q$ and $A$ are arrays of `Float64` (read from main memory)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Performance that matters

Not yet convinced?

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Performance comparison between the pseudo-transient (PT) and direct-iterative (DI) method resolving
2D shear-band formation out of a random noise cohesion field.

![pt_plastic2d](./figures/pt_plastic2d.png)

Räss et al. (2022) - https://doi.org/10.1029/2019GC008531
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Performance that matters - an example

Not yet convinced? Let's create have a look as an example.

Let's assess how close from memory copy (1355 GB/s) we can get solving a 2D diffusion problem on an Nvidia Tesla A100 GPU.

$$ ∇⋅(D ∇ C) = \frac{∂C}{∂t} $$

"""
using CUDA,BenchmarkTools
using ParallelStencil,ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(CUDA,Float64,2)
CUDA.device!(7) # select specific GPU
nx = ny = 512*64
C  = @rand(nx,ny)
D  = @rand(nx,ny)
dx = dy = dt = rand(); C2 = copy(T)
@parallel function diffusion_step!(C2, C, D, dt, dx, dy)
    @inn(T2) = @inn(T) + dt*@inn(D)*(@d2_xi(T)/dx/dx + @d2_yi(T)/dy/dy)
    return
end

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
We can now sample the performance on the GPU:
"""
t_it = @belapsed begin @parallel diffusion_step!($C2, $C, $D, $dt, $dx, $dy); end
T_eff = (2*1+1)*1/1e9*nx*ny*sizeof(Float64)/t_it
println("T_eff = $(T_eff) GiB/s using ParallelStencil on Nvidia A100 GPU")
println("So that's cool. We are getting close to hardware limit, running at $(T_eff_psind/1355) % of memory copy! 🚀")

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Why to still bother with GPU computing in 2022

Because it is still challenging

"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Why?
- Very few software uses it efficiently
- It requires to rethink the solving strategy as non-local operations will kill the fun
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## That's what we address in this workshop
- We show that both forward and inverse solvers can efficiently run on GPUs
- We demonstrate it by making an inversion of shear-driven Stokes flow
- We develop all this using the Julia language as it solves the "two-language problem"

"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Outline
1. The accelerated pseudo-transient (PT) method - a physics-motivated explanation
2. Application of the PT method to GPU supercomputing
3. Revisiting the adjoint method implementation using automatic differentiation (AD) and the PT method
4. Application: Point-wise inversion for power-law prefactor in a free-surface channel flow
5. Outlook and conclusion
6. Curious? Some useful resources
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 1. The accelerated pseudo-transient (PT) method
A physics-motivated explanation

Ivan
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 2. Application of the PT method to GPU supercomputing
Resolving nonlinear mechanical problems with elasto-viscoplastic rheology in 3D

![pt_plastic3d](./figures/pt_plastic3d.png)

Räss et al. (2022) - https://doi.org/10.1029/2019GC008531
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Scalability of the accelerated PT method I

Iteration count normalised by number of grid points in x-direction to remain mostly constant as function of `nx`.

![pt_iter_scale](./figures/pt_iter_scale.png)

Räss et al. (2022) - https://doi.org/10.1029/2019GC008531
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Scalability of the accelerated PT method II

Next, as the PT algorithm is fully local, we achieve ideal parallel efficiency on 2000 GPUs.

![pt_multigpu](./figures/pt_multigpu.png)

We use asynchronous GPU function execution to hide MPI communication behind computations. A ready.to-use feature in [`ImplicitGlobalGrid.jl`](https://github.com/eth-cscs/ImplicitGlobalGrid.jl).

Räss et al. (2022) - https://doi.org/10.1029/2019GC008531
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 3. Revisiting the adjoint method implementation
Using automatic differentiation (AD) and the PT method

Ivan
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## The adjoint method

![adjoint_inv](./figures/adjoint_inv.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## Automatic differentiation and the PT method

![adjoint_pt_ad](./figures/adjoint_pt_ad.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## AD approach in Julia

![adjoint_julia_tools](./figures/adjoint_julia_tools.png)
"""


#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 4. Application
Point-wise inversion for power-law prefactor in a free-surface channel flow

Inversion for viscosity in a free-surface channel flow

![inversion](./figures/inversion.gif)
"""


#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 5. Outlook and conclusion


"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## 6. Curious?
Some useful resources



"""








