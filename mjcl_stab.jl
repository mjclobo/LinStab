module LinStab

using FFTW, LinearAlgebra, Statistics, Peaks

## define stability functions

function lin_stab(U::Vector{Float64},V::Vector{Float64},H,beta,eta,Nx::Int64,Ny::Int64,rho::Vector{Float64},f0::Float64,g,Lx::Float64,Ly::Float64;rigid_lid=true)
    # U: (Nx x Nz) vector of zonal mean background velocity
    # V: (Ny x Nz) vector of meridional mean background velocity
    # beta: 
    # eta: The dimensional slope of the meridionally-sloping topography..for now

    Nz = length(rho)

    # eta gets factor of f/H in S definition function! 

    # define wavenumbers
    k_x = reshape(fftfreq(Nx, 2π/Lx*Nx),(1,Nx))
    k_y = reshape(fftfreq(Ny, 2π/Ly*Ny),(1,Ny))

    N_x = Nx
    N_xr = round(Int,Nx/2)
    k_xr = k_x[1:N_xr]

    # k_x,k_y,k_xr = wavenumber_grid(Nx,Ny,Lx,Ly)

    # k2 = k_x.^2 + k_y.^2            # this is for an isotropic wavenumber grid only (i.e. Nx=Ny)

    # define stretching matrix
    S = calc_stretching_mat(Nz,rho,f0,H,rho[1],g,rigid_lid,eta)

    # change dimensions of U and V to match domain size
    U2 = zeros(1,Nz); U2[:] = U; U2 = repeat(U2,outer=(Ny,1))
    U = zeros(1,Ny,Nz); U[1,:,:] = U2

    V2 = zeros(1,Nz); V2[:] = V; V2 = repeat(V2,outer=(N_xr,1))
    V = zeros(1,N_xr,Nz); V[1,:,:] = V2

    # define background QG PV gradient
    Qy = calc_PV_grad_y(U,beta,(f0/H[end])*eta,Ny,Nz,k_y,S)
    Qx = calc_PV_grad_x(V,0,N_xr,Nz,k_xr,S)

    # perform linear stability analysis
    evecs_all,evals_all = calc_lin_stab(Qy,Qx,U,V,S,k_xr,k_y,N_xr,Ny,Nz)

    # keep largest growth rates per wavenumber
    evecs,evals,max_evec_mag,max_evec_phase,max_eval = find_growth(evecs_all,evals_all,N_xr,Ny,Nz)

    # def rad
    evals_S = eigvals(-S); evecs_S = eigvecs(-S)
    sort_ind = sortperm(abs.(evals_S))

    # T2 = eltype(eigvals)
    r_d = Complex.(zeros(Nz,1))
    r_d[1] = sqrt(g*sum(H))/f0
    r_d[2:end] = @. sqrt(Complex(evals_S[sort_ind[2:end]]))^-1

    # normalizing eigenvectors
    norm_fact = sqrt.(sum(H) ./ (H .* sum( evecs_S .* evecs_S, dims=1)))
    evecs_S = norm_fact .* evecs_S

    return fftshift(evecs,[2]),fftshift(evals,[2]),max_evec_mag,max_evec_phase,max_eval,fftshift(k_x),fftshift(k_y),mean(Qx[1,:,:],dims=1),mean(Qy[1,:,:],dims=1),r_d,evecs_S
end

# function lin_stab(U::Vector{Float64},V::Vector{Float64},H,beta,eta,Nx::Int64,Ny::Int64,rho::Vector{Float64},f0::Float64,g,Lx::Float64,Ly::Float64,Qy)
#     # Takes Qy as arg; to mesh with GFJL output
#     # U: (Nx x Nz) vector of zonal mean background velocity
#     # V: (Ny x Nz) vector of meridional mean background velocity
#     # beta: 
#     # 

#     Nz = length(rho)
#     # define wavenumbers
#     k_x = reshape(fftfreq(Nx, 2π/Lx*Nx),(1,Nx))
#     k_y = reshape(fftfreq(Ny, 2π/Ly*Ny),(1,Ny))

#     # k_x,k_y = wavenumber_grid(Nx,Ny,Lx,Ly)

#     # k2 = k_x.^2 + k_y.^2            # this is for an isotropic wavenumber grid only (i.e. Nx=Ny)

#     # define stretching matrix
#     S = calc_stretching_mat(Nz,rho,f0,H,rho[1],g)

#     # change dimensions of U and V to match domain size
#     U2 = zeros(1,Nz); U2[:] = U; U2 = repeat(U2,outer=(Ny,1))
#     U = zeros(1,Ny,Nz); U[1,:,:] = U2

#     V2 = zeros(1,Nz); V2[:] = V; V2 = repeat(V2,outer=(Nx,1))
#     V = zeros(1,Nx,Nz); V[1,:,:] = V2

#     # define background QG PV gradients (TEMPORARY)
#     Qy = reshape(Qy[1,:,:],(1,Ny,Nz))
#     Qx = zeros(size(Qy))

#     # perform linear stability analysis
#     evecs_all,evals_all = calc_lin_stab(Qy,Qx,U,V,S,k_x,k_y,Nx,Ny,Nz)

#     # keep largest growth rates per wavenumber
#     evecs,evals,max_evec,max_eval = find_growth(evecs_all,evals_all,Nx,Ny,Nz)

#     # def rad
#     r_d = sqrt(gp(rho[1:2],rho[1],g)*H[1])/f0

#     return fftshift(evecs),fftshift(evals),max_evec,max_eval,fftshift(k_x),fftshift(k_y),mean(Qx[1,:,:],dims=1),mean(Qy[1,:,:],dims=1)
# end

function wavenumber_grid(Nx,Ny,Lx,Ly)
    #
    # nk_x = div(Nx,2)+1; nk_y = div(Ny,2)+1

    nk_x = Nx; nk_y = Ny

    nkr_x = round(Int,Nx/2)

    # k_x = reshape(LinRange(-2*pi/Lx*nk_x,2*pi/Lx*nk_x,nk_x),(1,Nx))
    # k_y = reshape(LinRange(-2*pi/Ly*nk_y,2*pi/Ly*nk_y,nk_y),(1,Ny))

    k_x = (fftfreq(Nx, 2π/Lx*Nx))
    k_y = (fftfreq(Ny, 2π/Ly*Ny))

    k_xr = k_x[1:nkr_x]

    return k_x,k_y,k_xr
end

function calc_PV_grad_y(U,beta,eta,Ny::Int64,Nz::Int64,k_y,S)
    # calculates PV gradients in one meridional direction
    # U is (Ny x Nz)
    # k_y is (Ny x 1)
    # 

    # Uyy = real.(ifft(-k_y.^2 .* fft(U)))

    # Uyy = repeat(Uyy, outer=(Nx, 1, 1))

    # Q_y = zeros(Nx,Nz)

    F = zeros(size(U))
    for i=1:Ny
        F[1,i,:] = S * U[1,i,:]
    end

    # Q_y = beta .- (Uyy .+ F)
    Q_y = beta .- F

    Q_y[:,:,end] = Q_y[:,:,end] .+ eta

    return Q_y
end

function calc_PV_grad_x(V,eta,Nx::Int64,Nz::Int64,k_x,S)
    # calculates PV gradients in one zonal direction

    # Vxx = real.(ifft(k_x.^2 .* fft(V)))

    # Q_x = zeros(Nx,Nz)

    F = zeros(size(V))
    for i=1:Nx
        F[1,i,:] = S * V[1,i,:]
    end

    # Q_x = Vxx .+ F

    Q_x = F

    return Q_x
end

function gp(rho,rho0,g)
    # g_prime = g*(rho[2]-rho[1])/rho0
    g_prime = g*(rho[2]-rho[1])/rho0
    return g_prime
end

function calc_stretching_mat(Nz,rho,f0,H,rho0,g,rigid_lid,eta)
    #
    S = zeros((Nz,Nz,))

    if rigid_lid
        alpha = 0
    else
        alpha = -f0^2/g/H[1]
    end

    eta_b_x = (f0/H[Nz])*eta

    S[1,1] = -f0^2/H[1]/gp(rho[1:2],rho0,g) + alpha
    S[1,2] = f0^2/H[1]/gp(rho[1:2],rho0,g)
    
    for i = 2:Nz-1
        S[i,i-1] = f0^2/H[i]/gp(rho[i-1:i],rho0,g)
        S[i,i]   = -(f0^2/H[i]/gp(rho[i-1:i],rho0,g) + f0^2/H[i]/gp(rho[i:i+1],rho0,g))
        S[i,i+1] = f0^2/H[i]/gp(rho[i:i+1],rho0,g)
    end

    S[Nz,Nz-1] = f0^2/H[Nz]/gp(rho[Nz-1:Nz],rho0,g)
    S[Nz,Nz]   = -f0^2/H[Nz]/gp(rho[Nz-1:Nz],rho0,g) + eta_b_x

    return S
end

function calc_lin_stab(Qy,Qx,U,V,S,k_x,k_y,Nx,Ny,Nz)
    Nx = length(k_x)
    evecs = zeros(Nx,Ny,Nz,Nz) .+ 0im
    evals = zeros(Nx,Ny,Nz) .+ 0im
    k2    = zeros(Nx,Ny)

    for i=1:Nx
        for j=1:Ny
            if i==1 && j==1
                # do nothing
            else
                A = make_diag(k_x[i] .* U[:,j,:] .+ k_y[j] .* V[:,i,:]) 

                ell = (S - (k_x[i]^2 + k_y[j]^2) * I)
                A2 = ell * A
                ell_i = inv(ell)
                Q2 = (k_x[i] * Qy[:,j,:] - k_y[j] * Qx[:,i,:]) .+ 0im
                B2 = transpose(ell_i) * transpose(A2 .+ make_diag(Q2)) .+ 0im

                k2[i,j] = (k_x[i]^2 + k_y[j]^2)

                eigB2 = eigen(B2)
                evecs[i,j,:,:] = eigB2.vectors
                evals[i,j,:] = eigB2.values

            end

        end
    end

    return evecs,evals
end

function make_diag(array_in)
    matrix_out = zeros(length(array_in),length(array_in))
    for i=eachindex(array_in)
        matrix_out[i,i] = array_in[i]
    end
    return matrix_out
end

function find_growth(evecs_all,evals_all,Nx,Ny,Nz)
    # for LSA
    
    evecs = zeros(Nx,Ny,Nz) .+ 0im; evals = zeros(Nx,Ny) .+ 0im

    for i=1:Nx
        for j=1:Ny
            indMax       = argmax(imag(evals_all[i,j,:]))       # find max growth amongst the Nz modes at a given wavenumber..why
            evals[i,j]   = evals_all[i,j,indMax]
            evecs[i,j,:] = evecs_all[i,j,:,indMax]
        end
    end

    sigma = imag(evals)

    # k_x[60] = -k_x[6] for i = j = 3 w.r.t. alpha gamma

    indMax = argmax(sigma)

    max_eval = sigma[indMax]

    max_evec_mag = abs.(evecs[indMax,:])
    max_evec_phase = atan.(imag(evecs[indMax,:]),real(evecs[indMax,:]))

    return evecs,evals,max_evec_mag,max_evec_phase,max_eval
end

"""
    calc_growth(t, E_in)
Estimate the growth rate of instabilities from a least-squares fit to energy
histories, where E_in = [KE1 KE2 KE3 PE32 PE52]
MODIFIED FROM @APALOCZY
"""
function calc_growth(t, E_in)
  E_tot = sum(E_in,dims=2)
  min_ind,trash = Peaks.findminima(diff(E_tot[:],dims=1))
  f = min_ind[end] + 1 # get first time index from the last renormalization cycle
  t, KE1_new, KE2_new, KE3_new, PE32_new, PE52_new = t[f:end], E_in[f:end,1], E_in[f:end,2], E_in[f:end,3], E_in[f:end,4], E_in[f:end,5]     # construct time series from last section of growth in upper layer
  
  sigma_KE1  = growth_rate(t,KE1_new)
  sigma_KE2  = growth_rate(t,KE2_new)
  sigma_KE3  = growth_rate(t,KE3_new)
  sigma_PE32 = growth_rate(t,PE32_new)
  sigma_PE52 = growth_rate(t,PE52_new)

  return [sigma_KE1, sigma_KE2, sigma_KE3, sigma_PE32, sigma_PE52]
end

function growth_rate(t,E)

    n = size(t)[1]
    d = Matrix(reshape(log.(E), (1, n)))
    gm = Matrix(reshape(t, (1, n)))
    Gm = Matrix([ones(n, 1) gm'])
    GmT = Gm'
    mv = inv(GmT*Gm)*(GmT*d')
    sigma = mv[2]/2

    return sigma
end


function find_kx_psi_vert(k_x,evecs,)

end

end # (module)

