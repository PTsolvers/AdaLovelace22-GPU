#src # This is needed to make this run as normal Julia file
using Markdown #src

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
# Parallel high-performance stencil computations on xPUs

Ludovic Räss & Ivan Utkin

_ETH Zurich_

![gpu](./figures/logo2.png)
"""

#src #########################################################################
#nb # %% A slide [markdown] {"slideshow": {"slide_type": "slide"}}
md"""
## The nice to have features

Wouldn't it be nice to have single code that:
- runs both on CPUs and GPUs (xPUs)?
- one can use for prototyping and production?
- runs at optimal performance?
"""

#nb # %% A slide [markdown] {"slideshow": {"slide_type": "fragment"}}
md"""
Hold on 🙂 ...
"""

