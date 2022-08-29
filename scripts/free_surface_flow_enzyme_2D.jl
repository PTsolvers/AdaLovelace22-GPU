using Plots,Printf
using Plots.PlotMeasures
using Enzyme

@views av(A)   = 0.5.*(A[1:end-1] .+ A[2:end])
@views avy(A)  = 0.5.*(A[1:end-1,:] .+ A[2:end,:])
@views avz(A)  = 0.5.*(A[:,1:end-1] .+ A[:,2:end])
@views av4(A)  = 0.25.*(A[1:end-1,1:end-1] .+ A[1:end-1,2:end] .+ A[2:end,1:end-1] .+ A[2:end,2:end])
@views bc2!(A) = (A[[1,end],:] .= A[[2,end-1],:]; A[:,[1,end]] .= A[:,[2,end-1]])

macro eII_xy()  esc(:( sqrt.((diff(vx[:,2:end-1],dims=1)./dy).^2 .+ (av4(diff(vx,dims=2)./dz)).^2)   )) end
macro eII_xz()  esc(:( sqrt.((av4(diff(vx,dims=1)./dy)).^2 .+ (diff(vx[2:end-1,:],dims=2)./dz).^2)   )) end
macro ηeff_xy() esc(:( 1.0./(1.0./(k0.*@eII_xy().^(npow-1.0)) .+ 1.0/ηreg)                           )) end
macro ηeff_xz() esc(:( 1.0./(1.0./(k0.*@eII_xz().^(npow-1.0)) .+ 1.0/ηreg)                           )) end

macro eII()     esc(:( sqrt.((avz(diff(vx,dims=1)./dy)).^2 .+ (avy(diff(vx,dims=2)./dz)).^2)         )) end
macro ηeff()    esc(:( 1.0./(1.0./(k0.*av4(@eII()).^(npow-1.0)) .+ 1.0/ηreg)                         )) end

macro ηeffτ()   esc(:( max.(ηeff_xy[1:end-1,:],ηeff_xy[2:end,:],ηeff_xz[:,1:end-1],ηeff_xz[:,2:end]) )) end
macro τxy()     esc(:( @ηeff_xy().*diff(vx[:,2:end-1],dims=1)./dy )) end
macro τxz()     esc(:( @ηeff_xz().*diff(vx[2:end-1,:],dims=2)./dz )) end

@views function residual!(r_vx,vx,k0,npow,ηreg,ρg,sinα,dy,dz)
    r_vx .= diff(@τxy(),dims=1)./dy .+ diff(@τxz(),dims=2)./dz .+ ρg*sinα
    return
end

@views function main()
    # physics
    # non-dimensional
    npow    = 1.0/3.0
    sinα    = sin(π/12)
    # dimensionally independent
    ly,lz   = 1.0,1.0 # [m]
    k0      = 1.0     # [Pa*s^npow]
    ρg      = 1.0     # [Pa/m]
    # scales
    psc     = ρg*lz
    ηsc     = psc*(k0/psc)^(1.0/npow)
    # dimensionally dependent
    ηreg    = 1e4*ηsc
    # numerics
    nz      = 100
    ny      = ceil(Int,nz*ly/lz)
    cfl     = 1/2.1
    ϵtol    = 1e-6
    ηrel    = 1e-2
    maxiter = 200max(ny,nz)
    ncheck  = 5max(ny,nz)
    re      = π/6
    # preprocessing
    dy,dz   = ly/ny,lz/nz
    yc,zc   = LinRange(-ly/2+dy/2,ly/2-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    vdτ     = cfl*min(dy,dz)
    # init
    vx      = zeros(ny  ,nz  )
    r_vx    = zeros(ny-2,nz-2)
    ηeff_xy = zeros(ny-1,nz-2)
    ηeff_xz = zeros(ny-2,nz-1)
    τxy     = zeros(ny-1,nz-2)
    τxz     = zeros(ny-2,nz-1)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        ηeff_xy              .= ηeff_xy.*(1.0-ηrel) .+ ηrel.*@ηeff_xy()
        ηeff_xz              .= ηeff_xz.*(1.0-ηrel) .+ ηrel.*@ηeff_xz()
        τxy                 .+= (.-τxy .+ ηeff_xy.*diff(vx[:,2:end-1],dims=1)./dy)./(1.0 + 2cfl*ny/re)
        τxz                 .+= (.-τxz .+ ηeff_xz.*diff(vx[2:end-1,:],dims=2)./dz)./(1.0 + 2cfl*ny/re)
        vx[2:end-1,2:end-1] .+= (diff(τxy,dims=1)./dy .+ diff(τxz,dims=2)./dz .+ ρg*sinα).*(vdτ*lz/re)./@ηeffτ()
        vx[:,end] .= vx[:,end-1]; vx[1,:] .= vx[2,:]
        if iter % ncheck == 0
            residual!(r_vx,vx,k0,npow,ηreg,ρg,sinα,dy,dz)
            err = maximum(abs.(r_vx))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = heatmap(yc,zc,vx'  ;aspect_ratio=1,xlabel="y",ylabel="z",title="Vx",xlims=(-ly/2,ly/2),ylims=(0,lz),right_margin=10mm)
            p2 = heatmap(yc[2:end-1],zc[2:end-1],(@ηeff())';aspect_ratio=1,xlabel="y",ylabel="z",title="ηeff",xlims=(-ly/2,ly/2),ylims=(0,lz),colorbar_scale=:log10)
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            display(plot(p1,p2,p3;size=(1200,400),layout=(1,3),bottom_margin=10mm,left_margin=10mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
