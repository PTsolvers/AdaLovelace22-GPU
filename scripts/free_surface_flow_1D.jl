using Plots,Printf
using Plots.PlotMeasures

@views av(A) = 0.5.*(A[1:end-1] .+ A[2:end])

@views function main()
    # physics
    # non-dimensional
    npow    = 1.0/3.0
    sinα    = sin(π/12)
    # dimensionally independent
    lz      = 1.0 # [m]
    k0      = 1.0 # [Pa*s^npow]
    ρg      = 1.0 # [Pa/m]
    # scales
    psc     = ρg*lz
    ηsc     = psc*(k0/psc)^(1.0/npow)
    # dimensionally dependent
    ηreg    = 1e4*ηsc
    # numerics
    nz      = 200
    cfl     = 1/1.1
    ϵtol    = 1e-6
    ηrel    = 1e-2
    maxiter = 200nz
    ncheck  = 5nz
    re      = π/12
    # preprocessing
    dz      = lz/nz
    zc      = LinRange(dz/2,lz-dz/2,nz)
    zv      = av(zc)
    vdτ     = cfl*dz
    # init
    vx      = zeros(nz)
    ηeff    = zeros(nz-1)
    τxz     = zeros(nz-1)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        ηeff         .= ηeff.*(1.0-ηrel) .+ ηrel./(1.0./(k0.*abs.(diff(vx)./dz).^(npow-1.0)) .+ 1.0/ηreg)
        τxz         .+= (.-τxz .+ ηeff.*diff(vx)./dz)./(1.0 + cfl*nz/re)
        vx[2:end-1] .+= (diff(τxz)./dz .+ ρg*sinα).*(vdτ*lz/re)./av(ηeff)
        vx[end]       = vx[end-1]
        if iter % ncheck == 0
            err = maximum(abs.(diff(τxz)./dz .+ ρg*sinα))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = plot(vx,zc;xlabel="Vx",ylabel="z",framestyle=:box,legend=false)
            p2 = plot(ηeff,zv;xlabel="ηeff",ylabel="z",xscale=:log10,framestyle=:box,legend=false)
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            display(plot(p1,p2,p3;size=(1200,400),layout=(1,3),bottom_margin=5mm,left_margin=5mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
