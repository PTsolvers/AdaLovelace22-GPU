using Plots,Printf
using Plots.PlotMeasures
using Enzyme

@inline hmean(a,b) = 1.0/(1.0/a + 1.0/b)

macro ∂vx_∂y(iy,iz) esc(:( (vx[$iy+1,$iz] - vx[$iy,$iz])/dy )) end
macro ∂vx_∂z(iy,iz) esc(:( (vx[$iy,$iz+1] - vx[$iy,$iz])/dz )) end

macro ∂vx_∂y_a4(iy,iz) esc(:( 0.25*(@∂vx_∂y($iy,$iz) + @∂vx_∂y($iy+1,$iz) + @∂vx_∂y($iy,$iz+1) + @∂vx_∂y($iy+1,$iz+1)) )) end
macro ∂vx_∂z_a4(iy,iz) esc(:( 0.25*(@∂vx_∂z($iy,$iz) + @∂vx_∂z($iy+1,$iz) + @∂vx_∂z($iy,$iz+1) + @∂vx_∂z($iy+1,$iz+1)) )) end

macro ∂vx_∂y_ay(iy,iz) esc(:( 0.5*(@∂vx_∂y($iy,$iz) + @∂vx_∂y($iy+1,$iz)) )) end
macro ∂vx_∂z_az(iy,iz) esc(:( 0.5*(@∂vx_∂z($iy,$iz) + @∂vx_∂z($iy,$iz+1)) )) end

macro τxy(iy,iz) esc(:( @ηeff_xy($iy,$iz)*@∂vx_∂y($iy,$iz+1) )) end
macro τxz(iy,iz) esc(:( @ηeff_xz($iy,$iz)*@∂vx_∂z($iy+1,$iz) )) end

macro eII_xy(iy,iz) esc(:( sqrt(@∂vx_∂y($iy,$iz+1)^2 + @∂vx_∂z_a4($iy,$iz)^2) )) end
macro eII_xz(iy,iz) esc(:( sqrt(@∂vx_∂y_a4($iy,$iz)^2 + @∂vx_∂z($iy+1,$iz)^2) )) end
macro eII_c(iy,iz)  esc(:( sqrt(@∂vx_∂y_ay($iy,$iz+1)^2 + @∂vx_∂z_az($iy+1,$iz)^2) )) end

macro ηeff_xy(iy,iz) esc(:( hmean(0.5*(k[$iy,$iz]+k[$iy,$iz+1])*@eII_xy($iy,$iz)^(npow-1.0), ηreg)                             )) end
macro ηeff_xz(iy,iz) esc(:( hmean(0.5*(k[$iy,$iz]+k[$iy+1,$iz])*@eII_xz($iy,$iz)^(npow-1.0), ηreg)                             )) end
macro ηeff_c(iy,iz)  esc(:( hmean(0.25*(k[$iy,$iz]+k[$iy,$iz+1]+k[$iy+1,$iz]+k[$iy+1,$iz+1])*@eII_c($iy,$iz)^(npow-1.0), ηreg) )) end

macro ηeffτ(iy,iz) esc(:( max(ηeff_xy[$iy,$iz],ηeff_xy[$iy+1,$iz],ηeff_xz[$iy,$iz],ηeff_xz[$iy,$iz+1]) )) end

function residual!(r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
    for iz = axes(r_vx,2), iy = axes(r_vx,1)
        r_vx[iy,iz] = (@τxy(iy+1,iz)-@τxy(iy,iz))/dy + (@τxz(iy,iz+1)-@τxz(iy,iz))/dz + ρgsinα
    end
    return
end

function ∂r_∂v!(JVP,Ψ,r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
    Enzyme.autodiff(residual!,Duplicated(r_vx,Ψ),Duplicated(vx,JVP),Const(k),Const(npow),Const(ηreg),Const(ρgsinα),Const(dy),Const(dz))
    return
end

function ∂r_∂k!(Jn,minus_Ψ,r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
    Enzyme.autodiff(residual!,Duplicated(r_vx,minus_Ψ),Const(vx),Duplicated(k,Jn),Const(npow),Const(ηreg),Const(ρgsinα),Const(dy),Const(dz))
    return
end

function eval_ηeff!(ηeff,ηeff_xy,ηeff_xz)
    for iz = axes(ηeff,2), iy = axes(ηeff,1) ηeff[iy,iz] = @ηeffτ(iy,iz) end
    return
end

@views function solve_forward!(vx,τxy,τxz,r_vx,k,ηeff_xy,ηeff_xz,ρgsinα,npow,ηreg,ηrel,psc,dy,dz,ny,nz,ly,lz,re,cfl,vdτ,ϵtol,maxiter,ncheck)
    vx      .= 0.0; r_vx    .= 0.0
    ηeff_xy .= 0.0; ηeff_xz .= 0.0
    τxy     .= 0.0; τxz     .= 0.0
    println("    forward solve:")
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        for iz = union(axes(τxy,2),axes(τxz,2)), iy = union(axes(τxy,1),axes(τxz,1))
            if iy ∈ axes(τxy,1) && iz ∈ axes(τxy,2)
                ηeff_xy[iy,iz] = ηeff_xy[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xy(iy,iz)
                τxy[iy,iz]    += (-τxy[iy,iz] + ηeff_xy[iy,iz]*@∂vx_∂y(iy,iz+1))/(1.0 + 2cfl*ny/re)
            end
            if iy ∈ axes(τxz,1) && iz ∈ axes(τxz,2)
                ηeff_xz[iy,iz] = ηeff_xz[iy,iz]*(1.0-ηrel) + ηrel*@ηeff_xz(iy,iz)
                τxz[iy,iz]    += (-τxz[iy,iz] + ηeff_xz[iy,iz]*@∂vx_∂z(iy+1,iz))/(1.0 + 2cfl*ny/re)
            end
        end
        for iz = axes(r_vx,2), iy = axes(r_vx,1)
            vx[iy+1,iz+1] += ((τxy[iy+1,iz]-τxy[iy,iz])/dy + (τxz[iy,iz+1]-τxz[iy,iz])/dz + ρgsinα)*(vdτ*lz/re)/@ηeffτ(iy,iz)
        end
        vx[:,end] .= vx[:,end-1]; vx[1,:] .= vx[2,:]
        if iter % ncheck == 0
            residual!(r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
            err = maximum(abs.(r_vx))*lz/psc
            @printf("      #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

@views function solve_adjoint!(Ψ,∂Ψ_∂τ,∂J_∂v,JVP,tmp,vx_obs,wt_cost,r_vx,vx,k,ηeff_xy,ηeff_xz,dτ,npow,ηreg,ρgsinα,dy,dz,ny,nz,dmp,ϵtol,maxiter,ncheck)
    for iz = axes(r_vx,2), iy = axes(r_vx,1)
        dτ[iy+1,iz+1] = 0.5*min(dy,dz)/sqrt(@ηeffτ(iy,iz))
    end
    dτ[[1,end],:] .= dτ[[2,end-1],:];  dτ[:,[1,end]] .=  dτ[:,[2,end-1]]
    ∂J_∂v .= (vx .- vx_obs).*wt_cost
    Ψ     .= 0.0
    println("  adjoint solve:")
    err = 2ϵtol; iter = 1
    while err >= ϵtol && iter <= maxiter
        JVP .= .-∂J_∂v; tmp .= Ψ[2:end-1,2:end-1]
        ∂r_∂v!(JVP,tmp,r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
        @. ∂Ψ_∂τ = ∂Ψ_∂τ*(1.0-dmp) + dτ*JVP
        @. Ψ    += dτ*∂Ψ_∂τ
        Ψ[1,:  ] .= Ψ[2,:    ]; Ψ[end,:] .= 0.0
        Ψ[:,end] .= Ψ[:,end-1]; Ψ[:  ,1] .= 0.0
        if iter % ncheck == 0
            err = maximum(abs.(JVP[2:end-1,2:end-1]))
            @printf("    #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

cost(vx,vx_obs,wt_cost) = 0.5*sum(wt_cost.*(vx.-vx_obs).^2)

function cost_gradient!(Jn,Ψ,minus_Ψ,r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
    Jn .= 0.0
    minus_Ψ .= .-Ψ[2:end-1,2:end-1]
    ∂r_∂k!(Jn,minus_Ψ,r_vx,vx,k,npow,ηreg,ρgsinα,dy,dz)
    Jn[[1,end],:] .= Jn[[2,end-1],:]; Jn[:,[1,end]] .= Jn[:,[2,end-1]]
    return
end

function smooth!(A,A2,nsm)
    for _ = 1:nsm
        A2[2:end-1,2:end-1] .= A[2:end-1,2:end-1] .+ 
            0.125.*(A[1:end-2,2:end-1] .+ A[3:end,2:end-1] .+ A[2:end-1,1:end-2] .+ A[2:end-1,3:end] .- 4.0.*A[2:end-1,2:end-1])
        A2[[1,end],:] .= A2[[2,end-1],:]; A2[:,[1,end]] .= A2[:,[2,end-1]]
        A,A2 = A2,A
    end
    return
end

function eval_ηeff!(ηeff,k,vx,ηreg,npow,dy,dz)
    for iz = axes(ηeff,2), iy = axes(ηeff,1)  ηeff[iy,iz] = @ηeff_c(iy,iz) end
    return
end

function make_plots(vx_synt,vx_inv,Jn,ηeff_inv,yc,zc,gd_iters,J_evo)
    opts_2d = (aspect_ratio=1,xlims=extrema(yc),ylims=extrema(zc),framestyle=:box)
    plots = (
        heatmap(yc,zc,vx_synt';xlabel="y",ylabel="z",title="vₓ synthetic",opts_2d...),
        heatmap(yc,zc,vx_inv' ;xlabel="y",ylabel="z",title="vₓ inverse"  ,opts_2d...),
        plot(yc,[vx_synt[:,end], vx_inv[:,end]];xlabel="y",ylabel="vₓ surface",framestyle=:box,label=["vₓ synthetic" "vₓ inverse"]),
        heatmap(yc[1:end-1],zc[1:end-1],Jn';xlabel="y",ylabel="z",title="dJ/dk",opts_2d...),
        heatmap(yc[2:end-1],zc[2:end-1],ηeff_inv' ;xlabel="y",ylabel="z",title="η inverse"  ,clims=(1e1,1e4),colorbar_scale=:log10,opts_2d...),
        plot(gd_iters,J_evo;xlabel="# gradient descent iters",ylabel="J/J_ini",yscale=:log10,framestyle=:box,legend=false,markershape=:circle),
    )
    display(plot(plots...;size=(1400,800),layout=(2,3),bottom_margin=10mm,left_margin=10mm,right_margin=10mm))
    return
end

@views function main()
    # physics
    # non-dimensional
    npow       = 1.0/3.0
    sinα       = sin(π/6)
    # dimensionally independent
    ly,lz      = 1.0,1.0 # [m]
    k0         = 1.0     # [Pa*s^npow]
    ρg         = 1.0     # [Pa/m]
    # scales
    psc        = ρg*lz
    ηsc        = psc*(k0/psc)^(1.0/npow)
    # dimensionally dependent
    ηreg       = 1e4*ηsc
    ρgsinα     = sinα*ρg
    # numerics
    nz         = 64
    ny         = ceil(Int,nz*ly/lz)
    cfl        = 1/2.1
    ϵtol       = 1e-6
    dmp        = 4/max(ny,nz)
    gd_ϵtol    = 1e-3
    ηrel       = 1e-2
    maxiter    = 200max(ny,nz)
    ncheck     = 10max(ny,nz)
    re         = π/10
    gd_maxiter = 500
    bt_maxiter = 10
    γ0         = 1e5
    nsm        = 100
    # preprocessing
    dy,dz      = ly/ny,lz/nz
    yc,zc      = LinRange(-ly/2+dy/2,ly/2-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    vdτ        = cfl*min(dy,dz)
    # allocate arrays
    vx_inv     = zeros(ny  ,nz  )
    vx_obs     = zeros(ny  ,nz  )
    r_vx       = zeros(ny-2,nz-2)
    ηeff_xy    = zeros(ny-1,nz-2)
    ηeff_xz    = zeros(ny-2,nz-1)
    ηeff_synt  = zeros(ny-2,nz-2)
    ηeff_inv   = zeros(ny-2,nz-2)
    τxy        = zeros(ny-1,nz-2)
    τxz        = zeros(ny-2,nz-1)
    k_synt     = zeros(ny-1,nz-1)
    k_inv      = zeros(ny-1,nz-1)
    Ψ          = zeros(ny  ,nz  )
    dτ         = zeros(ny  ,nz  )
    minusΨ     = zeros(ny-2,nz-2)
    ∂Ψ_∂τ      = zeros(ny  ,nz  )
    JVP        = zeros(ny  ,nz  )
    ∂J_∂v      = zeros(ny  ,nz  )
    tmp        = zeros(ny-2,nz-2)
    k_tmp      = zeros(ny-1,nz-1)
    k_ini      = zeros(ny-1,nz-1)
    Jn         = zeros(ny-1,nz-1)
    # init
    k_synt .= k0
    k_inv  .= k_synt.*(1.0 .+ 0.5.*(rand(ny-1,nz-1).-0.1))
    wt_cost = @. exp(5.0*(zc' - lz)/lz)
    # action
    # synthetic solution
    solve_forward!(vx_obs,τxy,τxz,r_vx,k_synt,ηeff_xy,ηeff_xz,ρgsinα,npow,ηreg,ηrel,psc,dy,dz,ny,nz,ly,lz,re,cfl,vdτ,ϵtol,maxiter,ncheck)
    eval_ηeff!(ηeff_synt,k_synt,vx_obs,ηreg,npow,dy,dz)
    # initial guess
    solve_forward!(vx_inv,τxy,τxz,r_vx,k_inv,ηeff_xy,ηeff_xz,ρgsinα,npow,ηreg,ηrel,psc,dy,dz,ny,nz,ly,lz,re,cfl,vdτ,ϵtol,maxiter,ncheck)
    γ     = γ0
    J_old = sqrt(cost(vx_inv,vx_obs,wt_cost)*dy*dz)
    J_ini = J_old
    J_evo = Float64[]; gd_iters = Int[]
    # gradient descent
    println("gradient descent:")
    for gd_iter = 1:gd_maxiter
        k_ini .= k_inv
        solve_adjoint!(Ψ,∂Ψ_∂τ,∂J_∂v,JVP,tmp,vx_obs,wt_cost,r_vx,vx_inv,k_inv,ηeff_xy,ηeff_xz,dτ,npow,ηreg,ρgsinα,dy,dz,ny,nz,dmp,ϵtol,maxiter,ncheck)
        cost_gradient!(Jn,Ψ,minusΨ,r_vx,vx_inv,k_inv,npow,ηreg,ρgsinα,dy,dz)
        # backtracking line search
        bt_iter = 1
        while bt_iter <= bt_maxiter
            println("  line search #iter $bt_iter:")
            δk = .-γ.*Jn
            @. k_inv = k_inv + δk
            smooth!(k_inv,k_tmp,nsm)
            solve_forward!(vx_inv,τxy,τxz,r_vx,k_inv,ηeff_xy,ηeff_xz,ρgsinα,npow,ηreg,ηrel,psc,dy,dz,ny,nz,ly,lz,re,cfl,vdτ,ϵtol,maxiter,ncheck)
            J_new = sqrt(cost(vx_inv,vx_obs,wt_cost)*dy*dz)
            if J_new < J_old
                γ = min(1.05*γ, 1e2*γ)
                J_old = J_new
                @printf("  new value accepted, misfit = %.1e\n", J_old)
                break
            else
                k_inv .= k_ini
                γ = max(0.5*γ, 1e-2*γ0)
                @printf("  restarting, new γ = %.3e\n",γ)
            end
            bt_iter += 1
        end
        # visualise
        push!(gd_iters,gd_iter); push!(J_evo,J_old/J_ini)
        eval_ηeff!(ηeff_inv,k_inv,vx_inv,ηreg,npow,dy,dz)
        make_plots(vx_obs,vx_inv,Jn,ηeff_inv,yc,zc,gd_iters,J_evo)
        # check convergence
        if bt_iter > bt_maxiter
            @printf("  line search couldn't descrease the misfit (%.1e)\n", J_old)
            break
        end
        if J_old/J_ini < gd_ϵtol
            @printf("gradient descent converged, misfit = %.1e\n", J_old)
            break
        else
            @printf("  gradient descent #iter = %d, misfit = %.1e\n", gd_iter, J_old)
        end
    end
    return
end

main()
