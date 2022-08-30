using Plots,Printf
using Plots.PlotMeasures
using Enzyme

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

function residual!(r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    for iz = axes(r_vx,2), iy = axes(r_vx,1)
        r_vx[iy,iz] = (@τxy(iy+1,iz)-@τxy(iy,iz))/dy + (@τxz(iy,iz+1)-@τxz(iy,iz))/dz + ρg*sinα
    end
    return
end

function ∂r_∂v!(JVP,vect,r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    Enzyme.autodiff(residual!,Duplicated(r_vx,vect),Duplicated(vx,JVP),Const(k),Const(npow),Const(ηreg),Const(ρg),Const(sinα),Const(dy),Const(dz))
    return
end

function eval_ηeff!(ηeff,ηeff_xy,ηeff_xz)
    for iz = axes(ηeff,2), iy = axes(ηeff,1) ηeff[iy,iz] = @ηeffτ(iy,iz) end
    return
end

@views function main()
    # physics
    # non-dimensional
    npow    = 1.0/3.0
    sinα    = sin(π/6)
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
    re      = π/10
    # preprocessing
    dy,dz   = ly/ny,lz/nz
    yc,zc   = LinRange(-ly/2+dy/2,ly/2-dy/2,ny),LinRange(dz/2,lz-dz/2,nz)
    vdτ     = cfl*min(dy,dz)
    # init
    vx      = zeros(ny  ,nz  )
    r_vx    = zeros(ny-2,nz-2)
    ηeff_xy = zeros(ny-1,nz-2)
    ηeff_xz = zeros(ny-2,nz-1)
    ηeff    = zeros(ny-2,nz-2)
    τxy     = zeros(ny-1,nz-2)
    τxz     = zeros(ny-2,nz-1)
    JVP     = zeros(ny-2,ny-2)
    vect    = zeros(ny-2,ny-2)
    k       = zeros(ny-1,nz-1)
    k      .= k0
    ∂r_∂v!(JVP,vect,r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
    # action
    iters_evo = Float64[]; errs_evo = Float64[]; err = 2ϵtol; iter = 1
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
            vx[iy+1,iz+1] += ((τxy[iy+1,iz]-τxy[iy,iz])/dy + (τxz[iy,iz+1]-τxz[iy,iz])/dz + ρg*sinα)*(vdτ*lz/re)/@ηeffτ(iy,iz)
        end
        vx[:,end] .= vx[:,end-1]; vx[1,:] .= vx[2,:]
        if iter % ncheck == 0
            residual!(r_vx,vx,k,npow,ηreg,ρg,sinα,dy,dz)
            eval_ηeff!(ηeff,ηeff_xy,ηeff_xz)
            err = maximum(abs.(r_vx))*lz/psc
            push!(iters_evo,iter/nz);push!(errs_evo,err)
            p1 = heatmap(yc,zc,vx';aspect_ratio=1,xlabel="y",ylabel="z",title="Vx",xlims=(-ly/2,ly/2),ylims=(0,lz),right_margin=10mm)
            # p2 = heatmap(yc[2:end-1],zc[2:end-1],r_vx';aspect_ratio=1,xlabel="y",ylabel="z",title="resid",xlims=(-ly/2,ly/2),ylims=(0,lz))
            p2 = heatmap(yc[2:end-1],zc[2:end-1],log10.(ηeff)';aspect_ratio=1,xlabel="y",ylabel="z",title="log10(ηeff)",xlims=(-ly/2,ly/2),ylims=(0,lz))
            p3 = plot(iters_evo,errs_evo;xlabel="niter/nx",ylabel="err",yscale=:log10,framestyle=:box,legend=false,markershape=:circle)
            p4 = plot(yc,vx[:,end];xlabel="y",ylabel="Vx",framestyle=:box,legend=false)
            display(plot(p1,p2,p3,p4;size=(800,800),layout=(2,2),bottom_margin=10mm,left_margin=10mm,right_margin=10mm))
            @printf("  #iter/nz=%.1f,err=%1.3e\n",iter/nz,err)
        end
        iter += 1
    end
    return
end

main()
