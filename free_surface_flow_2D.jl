using Plots,Printf
using Plots.PlotMeasures

@views av(A)   = 0.5.*(A[1:end-1] .+ A[2:end])
@views avy(A)  = 0.5.*(A[1:end-1,:] .+ A[2:end,:])
@views avz(A)  = 0.5.*(A[:,1:end-1] .+ A[:,2:end])
@views av4(A)  = 0.25.*(A[1:end-1,1:end-1] .+ A[2:end,1:end-1] .+ A[2:end,1:end-1] .+ A[2:end,2:end])
@views bc2!(A) = (A[[1,end],:] .= A[[2,end-1],:]; A[:,[1,end]] .= A[:,[2,end-1]])

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
    re      = π/7
    # preprocessing
    dy,dz   = ly/ny,lz/nz
    yv,zv   = LinRange(-ly/2,ly/2,ny+1),LinRange(0.0,lz,nz+1)
    yc,zc   = av(yv),av(zv)
    vdτ     = cfl*min(dy,dz)
    # init
    vx      = zeros(ny  ,nz  )
    eII     = zeros(ny-1,nz-1)
    ηeff    = zeros(ny,nz)
    τxz     = zeros(ny  ,nz-1)
    τxy     = zeros(ny-1,nz  )
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        eII                   .= sqrt.((avz(diff(vx,dims=1)./dy)).^2 .+ (avy(diff(vx,dims=2)./dz)).^2)
        ηeff[2:end-1,2:end-1] .= ηeff[2:end-1,2:end-1].*(1.0-ηrel) .+ ηrel./(1.0./(k0.*av4(eII).^(npow-1.0)) .+ 1.0/ηreg)
        bc2!(ηeff)
        τxy                  .+= (.-τxy .+ avy(ηeff).*diff(vx,dims=1)./dy)./(1.0 + 2cfl*ny/re)
        τxz                  .+= (.-τxz .+ avz(ηeff).*diff(vx,dims=2)./dz)./(1.0 + 2cfl*ny/re)
        vx[2:end-1,2:end-1]  .+= (diff(τxy[:,2:end-1],dims=1)./dy .+ diff(τxz[2:end-1,:],dims=2)./dz .+ ρg*sinα).*(vdτ*lz/re)./ηeff[2:end-1,2:end-1]
        vx[:,end]             .= vx[:,end-1]
        vx[1,:]               .= vx[2,:]
        if iter % ncheck == 0
            err = maximum(abs.(diff(τxy[:,2:end-1],dims=1)./dy .+ diff(τxz[2:end-1,:],dims=2)./dz .+ ρg*sinα))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = heatmap(yc,zc,vx'  ;aspect_ratio=1,xlabel="y",ylabel="z",title="Vx",xlims=(-ly/2,ly/2),ylims=(0,lz),right_margin=10mm)
            p2 = heatmap(yc,zc,ηeff';aspect_ratio=1,xlabel="y",ylabel="z",title="ηeff",xlims=(-ly/2,ly/2),ylims=(0,lz),colorbar_scale=:log10)
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            display(plot(p1,p2,p3;size=(1200,400),layout=(1,3),bottom_margin=10mm,left_margin=10mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
