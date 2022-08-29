using Plots,Printf
using Plots.PlotMeasures
using CUDA

@views av(A)   = 0.5.*(A[1:end-1] .+ A[2:end])
@views avy(A)  = 0.5.*(A[1:end-1,:] .+ A[2:end,:])
@views avz(A)  = 0.5.*(A[:,1:end-1] .+ A[:,2:end])
@views av4(A)  = 0.25.*(A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,1:end-1] .+ A[2:end,2:end])

macro avy(A)     esc(:( 0.5*($A[iy,iz] + $A[iy+1,iz]) )) end
macro avz(A)     esc(:( 0.5*($A[iy,iz] + $A[iy,iz+1]) )) end
macro av4(A)     esc(:( 0.25*($A[iy,iz] + $A[iy,iz+1] + $A[iy+1,iz] + $A[iy+1,iz+1]) )) end
macro d_ya(A)    esc(:( $A[iy+1,iz] - $A[iy,iz] )) end
macro d_za(A)    esc(:( $A[iy,iz+1] - $A[iy,iz] )) end
macro d_yi(A)    esc(:( $A[iy+1,iz+1] - $A[iy,iz+1] )) end
macro d_zi(A)    esc(:( $A[iy+1,iz+1] - $A[iy+1,iz] )) end

macro eII_xy()   esc(:( sqrt((@d_ya(vx)/dy)^2 + (0.25*(((vx[iy,iz+1] - vx[iy,iz])/dz) + ((vx[iy,iz+2] - vx[iy,iz+1])/dz) + ((vx[iy+1,iz+1] - vx[iy+1,iz])/dz) + ((vx[iy+1,iz+2] - vx[iy+1,iz+1])/dz)))^2) )) end
macro eII_xz()   esc(:( sqrt((@d_za(vx)/dz)^2 + (0.25*(((vx[iy+1,iz] - vx[iy,iz])/dy) + ((vx[iy+2,iz] - vx[iy+1,iz])/dy) + ((vx[iy+1,iz+1] - vx[iy,iz+1])/dy) + ((vx[iy+2,iz+1] - vx[iy+1,iz+1])/dy)))^2) )) end
macro ηeff_xy()  esc(:( 1.0/(1.0/(k0*@eII_xy()^(npow-1.0)) + 1.0/ηreg) )) end
macro ηeff_xz()  esc(:( 1.0/(1.0/(k0*@eII_xz()^(npow-1.0)) + 1.0/ηreg) )) end

# macro eII(iy,iz) esc(:( sqrt((0.5*((vx[$iy+1,$iz+1] - vx[$iy,$iz+1])/dy + (vx[$iy+1,$iz] - vx[$iy,$iz])/dy))^2 + (0.5*((vx[$iy+1,$iz+1] - vx[$iy+1,$iz])/dz + (vx[$iy,$iz+1] - vx[$iy,$iz])/dz))^2) )) end
# macro ηeff()     esc(:( 1.0/(1.0/(k0*(0.25*(@eII(ix,iy) + @eII(ix+1,iy) + @eII(ix,iy+1) + @eII(ix+1,iy+1)))^(npow-1.0)) + 1.0/ηreg) )) end
macro eII()      esc(:( sqrt.((avz(diff(vx,dims=1)./dy)).^2 .+ (avy(diff(vx,dims=2)./dz)).^2)         )) end
macro ηeff()     esc(:( 1.0./(1.0./(k0.*av4(@eII()).^(npow-1.0)) .+ 1.0/ηreg)                         )) end

macro ηeffτ()    esc(:( max(ηeff_xy[iy,iz],ηeff_xy[iy+1,iz],ηeff_xz[iy,iz],ηeff_xz[iy,iz+1]) )) end
macro τxy()      esc(:( @ηeff_xy()*@d_yi(vx)/dy )) end
macro τxz()      esc(:( @ηeff_xz()*@d_zi(vx)/dz )) end

function update_τ!(ηeff_xy,ηeff_xz,τxy,τxz,vx,k0,npow,ηreg,ηrel,re,ny,cfl,dy,dz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iy<=size(ηeff_xy,1) && iz<=size(ηeff_xy,2) ηeff_xy[iy,iz] = ηeff_xy[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xy() end
    if iy<=size(ηeff_xz,1) && iz<=size(ηeff_xz,2) ηeff_xz[iy,iz] = ηeff_xz[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xz() end
    if iy<=size(τxy,1)     && iz<=size(τxy,2)     τxy[iy,iz]     = τxy[iy,iz] + (-τxy[iy,iz] + ηeff_xy[iy,iz]*@d_yi(vx)/dy)/(1.0 + 2.0*cfl*ny/re) end
    if iy<=size(τxz,1)     && iz<=size(τxz,2)     τxz[iy,iz]     = τxz[iy,iz] + (-τxz[iy,iz] + ηeff_xz[iy,iz]*@d_zi(vx)/dz)/(1.0 + 2.0*cfl*ny/re) end
    return
end

function update_v!(vx,τxy,τxz,ηeff_xy,ηeff_xz,ρg,sinα,vdτ,lz,re,dy,dz)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iy<=size(vx,1)-2 && iz<=size(vx,2)-2
        vx[iy+1,iz+1] = vx[iy+1,iz+1] + (@d_ya(τxy)/dy + @d_za(τxz)/dz + ρg*sinα)*(vdτ*lz/re)/@ηeffτ() end
    return
end

function apply_bc!(vx)
    iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
    iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
    if iy<=size(vx,1) && iz==size(vx,2) vx[iy,iz] = vx[iy,iz-1] end
    if iy==1          && iz<=size(vx,2) vx[iy,iz] = vx[iy+1,iz] end
    return
end

# function residual!(r_vx,τxy,τxz,ρg,sinα,dy,dz)
#     iy = (blockIdx().x-1)*blockDim().x + threadIdx().x
#     iz = (blockIdx().y-1)*blockDim().y + threadIdx().y
#     if iy<=size(r_vx,1) && iz<=size(r_vx,2) r_vx[iy,iz] = (@d_ya(@τxy())/dy + @d_za(@τxz())/dz + ρg*sinα) end
#     return
# end

@views function residual2!(r_vx,τxy,τxz,ρg,sinα,dy,dz)
    r_vx .= diff(τxy,dims=1)./dy .+ diff(τxz,dims=2)./dz .+ ρg*sinα
    return
end

@views function main()
    # physics
    # non-dimensional
    npow     = 1.0/3.0
    sinα     = sin(π/12)
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
    re       = π/7
    # preprocessing
    dy,dz    = ly/ny,lz/nz
    yc,zc    = LinRange(-ly/2+dy/2,ly/2-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    yv,zv    = av(yc),av(zc)
    vdτ      = cfl*min(dy,dz)
    # init
    vx       = CUDA.zeros(Float64,ny  ,nz  )
    # r_vx     = CUDA.zeros(Float64,ny-2,nz-2)
    ηeff_xy  = CUDA.zeros(Float64,ny-1,nz-2)
    ηeff_xz  = CUDA.zeros(Float64,ny-2,nz-1)
    τxy      = CUDA.zeros(Float64,ny-1,nz-2)
    τxz      = CUDA.zeros(Float64,ny-2,nz-1)
    r_vx     = zeros(ny-2,nz-2)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        # CUDA.@sync @cuda threads=nthreads blocks=nblocks update_ηeff!(ηeff,vx,k0,npow,ηreg,ηrel,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_τ!(ηeff_xy,ηeff_xz,τxy,τxz,vx,k0,npow,ηreg,ηrel,re,ny,cfl,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks update_v!(vx,τxy,τxz,ηeff_xy,ηeff_xz,ρg,sinα,vdτ,lz,re,dy,dz)
        CUDA.@sync @cuda threads=nthreads blocks=nblocks apply_bc!(vx)
        if iter % ncheck == 0
            # CUDA.@sync @cuda threads=nthreads blocks=nblocks residual!(r_vx,τxy,τxz,ρg,sinα,dy,dz)
            residual2!(r_vx,Array(τxy),Array(τxz),ρg,sinα,dy,dz)
            err = maximum(abs.(r_vx))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = heatmap(yc,zc,Array(vx)'     ;aspect_ratio=1,xlabel="y",ylabel="z",title="Vx",xlims=(-ly/2,ly/2),ylims=(0,lz),right_margin=10mm)
            p2 = heatmap(yv,zv,Array(@ηeff())';aspect_ratio=1,xlabel="y",ylabel="z",title="ηeff",xlims=(-ly/2,ly/2),ylims=(0,lz),colorbar_scale=:log10)
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            display(plot(p1,p2,p3;size=(1200,400),layout=(1,3),bottom_margin=10mm,left_margin=10mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
