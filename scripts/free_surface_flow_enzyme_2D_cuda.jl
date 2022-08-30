using Plots,Printf
using Plots.PlotMeasures
using Enzyme
using CUDA

@inline hmean(a,b) = 1.0/(1.0/a + 1.0/b)

macro ∂vx_∂y(iy,iz) esc(:( (vx[$iy+1,$iz] - vx[$iy,$iz])/dy )) end
macro ∂vx_∂z(iy,iz) esc(:( (vx[$iy,$iz+1] - vx[$iy,$iz])/dz )) end

macro ∂vx_∂y_a4(iy,iz) esc(:( 0.25*(@∂vx_∂y($iy,$iz) + @∂vx_∂y($iy+1,$iz) + @∂vx_∂y($iy,$iz+1) + @∂vx_∂y($iy+1,$iz+1)) )) end
macro ∂vx_∂z_a4(iy,iz) esc(:( 0.25*(@∂vx_∂z($iy,$iz) + @∂vx_∂z($iy+1,$iz) + @∂vx_∂z($iy,$iz+1) + @∂vx_∂z($iy+1,$iz+1)) )) end

macro τxy(iy,iz) esc(:( @ηeff_xy($iy,$iz)*@∂vx_∂y($iy,$iz+1) )) end
macro τxz(iy,iz) esc(:( @ηeff_xz($iy,$iz)*@∂vx_∂z($iy+1,$iz) )) end

macro eII_xy(iy,iz) esc(:( sqrt(@∂vx_∂y($iy,$iz+1)^2 + @∂vx_∂z_a4($iy,$iz)^2) )) end
macro eII_xz(iy,iz) esc(:( sqrt(@∂vx_∂y_a4($iy,$iz)^2 + @∂vx_∂z($iy+1,$iz)^2) )) end

macro ηeff_xy(iy,iz) esc(:( hmean(0.5*(k[$iy,iz]+k[$iy,iz+1])*@eII_xy($iy,$iz)^(npow-1.0), ηreg) )) end
macro ηeff_xz(iy,iz) esc(:( hmean(0.5*(k[$iy,iz]+k[$iy+1,iz])*@eII_xz($iy,$iz)^(npow-1.0), ηreg) )) end

macro ηeffτ(iy,iz) esc(:( max(ηeff_xy[$iy,$iz],ηeff_xy[$iy+1,$iz],ηeff_xz[$iy,$iz],ηeff_xz[$iy,$iz+1]) )) end

@inbounds function residual!(r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iz <= size(r_vx,2) && iy <= size(r_vx,1)
        r_vx[iy,iz] = (@τxy(iy+1,iz)-@τxy(iy,iz))/dy + (@τxz(iy,iz+1)-@τxz(iy,iz))/dz + ρg*sinα
    end
    return
end

function ∂r_∂v!(JVP,vect,r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    Enzyme.autodiff_deferred(residual!,Duplicated(r_vx,vect),Duplicated(vx,JVP),Const(k),Const(npow),Const(ηreg),Const(ρg),Const(sinα),Const(dy),Const(dz))
    return
end

@inbounds function update_τ!(τxy,τxz,ηeff_xy,ηeff_xz,vx,k,npow,ηrel,ηreg,re,cfl,ny,dy,dz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iz <= size(τxy,2) && iy <= size(τxy,1)
        ηeff_xy[iy,iz] = ηeff_xy[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xy(iy,iz)
        τxy[iy,iz]     = τxy[iy,iz] + (-τxy[iy,iz] + ηeff_xy[iy,iz]*@∂vx_∂y(iy,iz+1))/(1.0 + 2cfl*ny/re)
    end
    if iz <= size(τxz,2) && iy <= size(τxz,1)
        ηeff_xz[iy,iz] = ηeff_xz[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xz(iy,iz)
        τxz[iy,iz]     = τxz[iy,iz] + (-τxz[iy,iz] + ηeff_xz[iy,iz]*@∂vx_∂z(iy+1,iz))/(1.0 + 2cfl*ny/re)
    end
    return
end

@inbounds function update_v!(vx,τxy,τxz,ηeff_xy,ηeff_xz,k,npow,ηreg,ρg,sinα,vdτ,lz,re,dy,dz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iz <= size(vx,2)-2 && iy <= size(vx,1)-2
        vx[iy+1,iz+1] = vx[iy+1,iz+1] + ((τxy[iy+1,iz]-τxy[iy,iz])/dy + (τxz[iy,iz+1]-τxz[iy,iz])/dz + ρg*sinα)*(vdτ*lz/re)/@ηeffτ(iy,iz)
    end
    return
end

@inbounds function apply_bc!(vx)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iz==size(vx,2) && iy<=size(vx,1) vx[iy,iz] = vx[iy,iz-1] end
    if iz<=size(vx,2) && iy==1          vx[iy,iz] = vx[iy+1,iz] end
    return
end

@inbounds function eval_ηeff!(ηeff,ηeff_xy,ηeff_xz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iz<=size(ηeff,2) && iy<=size(ηeff,1) ηeff[iy,iz] = @ηeffτ(iy,iz) end
    return
end

@views function main()
    # physics
    # non-dimensional
    npow     = 1.0/3.0
    sinα     = sin(π/6)
    # dimensionally independent
    ly,lz    = 1.0,1.0 # [m]
    k0       = 1.0     # [Pa*s^npow]
    ρg       = 1.0     # [Pa/m]
    # scales
    psc      = ρg*lz
    ηsc      = psc*(k0/psc)^(1.0/npow)
    # dimensionally dependent
    ηreg     = 1e4*ηsc
    # numerics
    nz       = 128
    ny       = ceil(Int,nz*ly/lz)
    nthreads = (16,16)
    nblocks  = cld.((ny,nz),nthreads)
    cfl      = 1/2.1
    ϵtol     = 1e-6
    ηrel     = 1e-2
    maxiter  = 200max(ny,nz)
    ncheck   = 5max(ny,nz)
    re       = π/10
    # preprocessing
    dy,dz    = ly/ny,lz/nz
    yc,zc    = LinRange(-ly/2+dy/2,ly/2-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    vdτ      = cfl*min(dy,dz)
    # init
    vx       = CUDA.zeros(Float64,ny  ,nz  )
    r_vx     = CUDA.zeros(Float64,ny-2,nz-2)
    ηeff_xy  = CUDA.zeros(Float64,ny-1,nz-2)
    ηeff_xz  = CUDA.zeros(Float64,ny-2,nz-1)
    ηeff     = CUDA.zeros(Float64,ny-2,nz-2)
    τxy      = CUDA.zeros(Float64,ny-1,nz-2)
    τxz      = CUDA.zeros(Float64,ny-2,nz-1)
    JVP      = CUDA.zeros(Float64,ny-2,nz-2)
    vect     = CUDA.zeros(Float64,ny-2,nz-2)
    k        = k0.*CUDA.ones(Float64,ny-1,nz-1)
    # Inverse (JVP)
    # ∂r_∂v!(JVP,vect,r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        # CUDA.@sync @cuda threads=nthreads blocks=nblocks update_ηeff!(ηeff,vx,k,npow,ηreg,ηrel,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_τ!(τxy,τxz,ηeff_xy,ηeff_xz,vx,k,npow,ηrel,ηreg,re,cfl,ny,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_v!(vx,τxy,τxz,ηeff_xy,ηeff_xz,k,npow,ηreg,ρg,sinα,vdτ,lz,re,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks apply_bc!(vx)
        if iter % ncheck == 0
            CUDA.@sync @cuda threads=nthreads blocks=nblocks residual!(r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
            CUDA.@sync @cuda threads=nthreads blocks=nblocks eval_ηeff!(ηeff,ηeff_xy,ηeff_xz)
            err = maximum(abs.(r_vx))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = heatmap(yc,zc,Array(vx)';aspect_ratio=1,xlabel="y",ylabel="z",title="Vx",xlims=(-ly/2,ly/2),ylims=(0,lz),right_margin=10mm)
            # p2 = heatmap(yc[2:end-1],zc[2:end-1],Array(r_vx)';aspect_ratio=1,xlabel="y",ylabel="z",title="resid",xlims=(-ly/2,ly/2),ylims=(0,lz))
            p2 = heatmap(yc[2:end-1],zc[2:end-1],Array(log10.(ηeff))';aspect_ratio=1,xlabel="y",ylabel="z",title="log10(ηeff)",xlims=(-ly/2,ly/2),ylims=(0,lz))
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            p4 = plot(yc,Array(vx)[:,end];xlabel="y",ylabel="Vx",framestyle=:box,legend=false)
            display(plot(p1,p2,p3,p4;size=(800,800),layout=(2,2),bottom_margin=10mm,left_margin=10mm,right_margin=10mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
