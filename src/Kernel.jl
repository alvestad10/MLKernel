export setKernel!

########################

abstract type Kernel end

abstract type KernelParameters end

abstract type ConstantKernelParameters <: KernelParameters end
abstract type AHOConstantKernelParameters <: ConstantKernelParameters end


function getKernelParams(pK::T) where {T <: AHOConstantKernelParameters}
    return pK.p
end

function getKernelParamsSim(::T) where {T <: AHOConstantKernelParameters}
    return []
end

function dK(u,p,::T) where {T <: AHOConstantKernelParameters}
    return nothing
end

function K_dK(u,p,pK::T) where {T <: AHOConstantKernelParameters}
    return K(u,p,pK),dK(u,p,pK)
end





"""
Constant kernel with the entries of the complex matrix K as the parameters.
We get the ``H`` by built in square root ``H=sqrt(K)``
"""
mutable struct AHO_ConstKernel_K <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_K(p)

    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]

    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_K(p,H,K)
end


function AHO_ConstKernel_K(M::AHO)
    @unpack t_steps = M.contour 
    
    p = Matrix([Diagonal(ones(t_steps)) ; zeros(t_steps,t_steps)])

    return AHO_ConstKernel_K(p)
end

function updateKernel!(pK::AHO_ConstKernel_K)
    
    _K = pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:]    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_K)
    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_K)
    _K = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end


"""
Constant kernel with the entries of the complex matrix H as the parameters.
We get the kernel by squaring ``K=H^2``
"""
mutable struct AHO_ConstKernel_H <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_H(p)
    sqrtK = p
    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
         - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
        
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
         + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    
    K = hcat([KRe;KIm],[-KIm;KRe])

    return AHO_ConstKernel_H(p,sqrtK,K)
end

function AHO_ConstKernel_H(M::AHO)
    @unpack t_steps = M.contour 
    kRe = Matrix( Diagonal(ones(t_steps))) 
    kIm = zeros(t_steps,t_steps)
    K = vcat(kRe,kIm)
    return AHO_ConstKernel(K)
end


function updateKernel!(pK::AHO_ConstKernel_H)
    @unpack p = pK
    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
         - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
        
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
         + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    

    pK.sqrtK = p
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,::AHO_ConstKernel_H)

    KRe = (transpose(p[1:div(end,2),:])*p[1:div(end,2),:]
    - transpose(p[div(end,2)+1:end,:])*p[div(end,2)+1:end,:])
    
    KIm = (transpose(p[1:div(end,2),:])*p[div(end,2)+1:end,:] 
    + transpose(p[div(end,2)+1:end,:])*p[1:div(end,2),:])
    

    return KRe,KIm
end


function sqrtK(u,p,::AHO_ConstKernel_H)
    return p
end





"""
Constant kernel with the entries of the matrix defined by the entries
``
H_{ij} = s_{ij}(\\cos(θ_{ij}) + \\sin(θ_{ij})
``
such that the parameters is the real matrices ``θ`` and ``s``. The kernel is
obtained by squaring:``K=H^2``.
"""
mutable struct AHO_ConstKernel_sincos <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_sincos(θ,s)
    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)

    sqrtK = [HRe;HIm]
    K = hcat([KRe;KIm],[-KIm;KRe])
    
    return AHO_ConstKernel_sincos([θ;s],sqrtK,K)
end

function ConstantKernel_sincos(M::AHO)
    @unpack t_steps = M.contour 
    θ = zeros(t_steps,t_steps)
    s = Matrix( Diagonal(ones(t_steps))) 
    
    return AHO_ConstKernel_sincos(θ,s)
end

function updateKernel!(pK::AHO_ConstKernel_sincos)

    θ = pK.p[1:div(end,2),:]
    s = pK.p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)

    pK.sqrtK = [HRe;HIm]
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,::AHO_ConstKernel_sincos)

    θ = p[1:div(end,2),:]
    s = p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)

    KRe = (transpose(HRe)*HRe
         - transpose(HIm)*HIm)
        
    KIm = (transpose(HRe)*HIm
         + transpose(HIm)*HRe)
    return KRe,KIm
end

function sqrtK(u,p,::AHO_ConstKernel_sincos)

    θ = p[1:div(end,2),:]
    s = p[div(end,2)+1:end,:]

    HRe = s .* cos.(θ)
    HIm = s .* sin.(θ)
    
    return [HRe;HIm]
end





"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = e^{iP}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiP <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiP(p)
    _K = exp(im*p)
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiP(p,H,K)
end

function AHO_ConstKernel_expiP(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiP(p)
end


function updateKernel!(pK::AHO_ConstKernel_expiP)

    _K = exp(im*pK.p)
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expiP)
    _K = exp(im*p)
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiP)
    _K = exp(im*p)
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end

"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = e^{A}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expA <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expA(p)
    
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expA(p,H,K)
end

function AHO_ConstKernel_expA(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(2t_steps,t_steps)

    return AHO_ConstKernel_expA(p)
end


function updateKernel!(pK::AHO_ConstKernel_expA)

    _K = exp(pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expA)
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    return real(_K),imag(_K)
end

function sqrtK(u,p,::AHO_ConstKernel_expA)
    _K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end

"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
H = e^{A}
```
We get ``K`` by the built in squaring ``H^2``.
"""
mutable struct AHO_ConstKernel_HexpA <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_HexpA(p)
    
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    H = [real(_H);imag(_H)]
    
    _K=_H^2
    #_K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    #_H = sqrt(_K)
    #H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_HexpA(p,H,K)
end

function AHO_ConstKernel_HexpA(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(2t_steps,t_steps)

    return AHO_ConstKernel_HexpA(p)
end


function updateKernel!(pK::AHO_ConstKernel_HexpA)

    _H = exp(pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:])
    pK.sqrtK = [real(_H);imag(_H)]
    
    _K=_H^2
    #_K = exp(pK.p[1:div(end,2),:] + im*pK.p[div(end,2)+1:end,:])
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    #_H = sqrt(_K)
    #pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_HexpA)
    #_K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    
    _K=_H^2
    return real(_K),imag(_K)
end

function sqrtK(u,p,::AHO_ConstKernel_HexpA)
    _H = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    

    #_K = exp(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:])
    #_H = sqrt(_K)
    return [real(_H);imag(_H)]    
end


"""
Constant kernel with the entries of the real matrix P such that the kernel becomes
```math
K = √(1/M) + e^{iP}
```
We get the kernel by the built in squaring ``K = H^2``.
"""
mutable struct AHO_ConstKernel_invM_H <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtinvM::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_invM_H(p,sqrtinvM)

    _H = sqrtinvM[1:div(end,2),:] + im*sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(p)))
    H = [real(_H);imag(_H)]
    
    _K = _H^2
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])
    
    return AHO_ConstKernel_invM_H(p,sqrtinvM,H,K)
end

function AHO_ConstKernel_invM_H(M::AHO; m=M.m, g=1)
    @unpack a,t_steps,κ = M.contour 
    A = zeros(Complex{typeof(m)},t_steps,t_steps)
    for j in 1:t_steps
        jm1 = mod1(j-1,t_steps)
        jp1 = mod1(j+1,t_steps)
        A[j,j] = g*((1/a[jm1]) + (1/a[j])) - 0.5*(a[jm1] + a[j])*m
        A[j,jp1] = -g*(1/a[j])
        A[j,jm1] = -g*(1/a[jm1])
    end
    A = -im*A
    sqrtK = sqrt(inv(A))
    p = zeros(Float64,t_steps,t_steps)
    
    return AHO_ConstKernel_invM_H(p,[real(sqrtK) ; imag(sqrtK)])
end

function updateKernel!(pK::AHO_ConstKernel_invM_H)
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(pK.p)))
    pK.sqrtK = [real(_H);imag(_H)]
    
    _K = _H^2
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])
end

function K(u,p,pK::AHO_ConstKernel_invM_H)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*Symmetric(p) .+ _ϵ)
    _K = _H^2
    return real(_K),imag(_K)
end


function sqrtK(u,p,pK::AHO_ConstKernel_invM_H)
    _H = pK.sqrtinvM[1:div(end,2),:] + im*pK.sqrtinvM[div(end,2)+1:end,:] + exp(im*parent(Symmetric(p)))
    _H = exp(im*p)
    return [real(_H);imag(_H)]
end



"""
Constant kernel with the entries of the Hermitian matrix P such that the kernel becomes
```math
K = e^{iP}
```
This will in practise mean that we are exponentiating an anti-HErmitian matatrix ``iP``.
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiHerm <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiHerm(p)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L)))
    
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiHerm(p,H,K)
end

function AHO_ConstKernel_expiHerm(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiHerm(p)
end

function updateKernel!(pK::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(pK.p) + Hermitian(im*pK.p,:L)))    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L)))   
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiHerm)
    _K = exp(im*(Symmetric(p) + Hermitian(im*p,:L))) 
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end


"""
Constant kernel with the entries of the symmetric matrix P such that the kernel becomes
```math
K = e^{iP}
```
We get ``H`` by the built in square root ``sqrt(K)``.
"""
mutable struct AHO_ConstKernel_expiSym <: AHOConstantKernelParameters
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function AHO_ConstKernel_expiSym(p)
    _K = exp(im*parent(Symmetric(p)))
    
    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_expiSym(p,H,K)
end

function AHO_ConstKernel_expiSym(M::AHO)
    @unpack t_steps = M.contour 
    
    p = zeros(t_steps,t_steps)

    return AHO_ConstKernel_expiSym(p)
end

function updateKernel!(pK::AHO_ConstKernel_expiSym)
    _K = exp(im*parent(Symmetric(pK.p)))    
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,pK::AHO_ConstKernel_expiSym)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _p = Symmetric(p) .+ _ϵ
    _K = exp(im*_p) 
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_expiSym)
    _ϵ = sparse([1,size(p)[1]], [size(p)[2],size(p)[2]], [1e-10,0.])
    _p = Symmetric(p) .+ _ϵ
    _K = exp(im*_p)
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end


"""
EXPERIMENTAL

Kernel represtented by a complex Gaussian kernel.
"""
mutable struct AHO_ConstKernel_Gauss <: AHOConstantKernelParameters
    model::AHO
    p::Matrix{Float64}
    sqrtK::Matrix{Float64}
    K::Matrix{Float64}
end

function GaussK(p,M)
    γ = p[1]
    vr=p[2]
    vrj = p[3]
    x0 = M.contour.x0[1:end-1]
    dx = x0 .- transpose(x0)
    μ = p[4:end]
    return  (vr^2+vrj^2)*exp.( -conj.(dx) .* (dx)/γ) .+
     im*vr*vrj*(exp.( -conj.(dx .- μ) .* (dx .- μ)/γ) .-
     exp.(-conj.(dx .+ μ) .* (dx .+ μ)/γ))
end

function AHO_ConstKernel_Gauss(p,M::Model)

    #_κR = p[1]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[5]) 
    #_κI = p[2]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[6]) 
    #_κRI = p[3]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[7]) 
    #_κIR = p[4]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[8]) 

    #_K = 0.5*(_κR + _κI + im*(_κRI - _κIR)) 
    _K = GaussK(p,M)

    KRe = real(_K)
    KIm = imag(_K)
    K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    H = [real(_H);imag(_H)]
    
    return AHO_ConstKernel_Gauss(M,Matrix(transpose(p)),H,K)
end

function AHO_ConstantKernel_Gaussian(M::AHO)
    @unpack t_steps = M.contour 
    
    p = [1.,1.,0.,zeros(Float64,t_steps)...]

    return AHO_ConstKernel_Gauss(p,M)
end


function updateKernel!(pK::AHO_ConstKernel_Gauss)
    p = pK.p
    M = pK.model
    #_κR = p[1]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[5]) 
    #_κI = p[2]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[6]) 
    #_κRI = p[3]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[7]) 
    #_κIR = p[4]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[8]) 

    #_K = 0.5*(_κR + _κI + im*(_κRI - _κIR)) 
    _K = GaussK(p,M)
    KRe = real(_K)
    KIm = imag(_K)
    pK.K = hcat([KRe;KIm],[-KIm;KRe])

    _H = sqrt(_K)
    pK.sqrtK = [real(_H);imag(_H)]
end

function K(u,p,pK::AHO_ConstKernel_Gauss)
    M = pK.model
    #_κR = p[1]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[5]) 
    #_κI = p[2]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[6]) 
    #_κRI = p[3]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[7]) 
    #_κIR = p[4]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[8]) 

    #_K = 0.5*(_κR + _κI + im*(_κRI - _κIR)) 
    _K = GaussK(p,M)
    return real(_K),imag(_K)
end


function sqrtK(u,p,::AHO_ConstKernel_Gauss)
    #M = pK.model
    #_κR = p[1]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[5]) 
    #_κI = p[2]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[6]) 
    #_κRI = p[3]*exp.( -(real(M.contour.x0[1:end-1]) .- transpose(imag(M.contour.x0[1:end-1]))).^2/p[7]) 
    #_κIR = p[4]*exp.( -(imag(M.contour.x0[1:end-1]) .- transpose(real(M.contour.x0[1:end-1]))).^2/p[8]) 

    #_K = 0.5*(_κR + _κI + im*(_κRI - _κIR)) 
    _K = GaussK(p,M)
    _H = sqrt(_K)
    return [real(_H);imag(_H)]    
end








##### LM_AHO Constant kernel
abstract type LM_AHOConstantKernelParameters <: ConstantKernelParameters end


function dK(u,p,::T) where {T <: LM_AHOConstantKernelParameters}
    return nothing
end

function updateKernel!(pK::T) where {T <: LM_AHOConstantKernelParameters}
    nothing
end

function getKernelParamsSim(pK::T) where {T <: LM_AHOConstantKernelParameters}
    return getKernelParams(pK)
end


function K_dK(u,p,pK::T) where {T <: LM_AHOConstantKernelParameters}
    return K(u,p,pK),dK(u,p,pK)
end


"""
Kernel defined by a complex number ``H``, and normalized to length ``1``
To get the kernel we square ``K=H^2``
"""
struct LM_AHO_ConstKernel_H <: LM_AHOConstantKernelParameters
    model::LM_AHO
    H
    function LM_AHO_ConstKernel(model::LM_AHO; H=nothing)
        if isnothing(H)
            H = [pi,0.]
        end
        new(model,H)
    end
end


function K(u,p,::LM_AHO_ConstKernel_H)
    _p = p ./ sqrt(p[1]^2 + p[2]^2)
    return _p[1]^2 - _p[2]^2, 2*_p[1]*_p[2]
end


function sqrtK(u,p,::LM_AHO_ConstKernel_H)
    return p ./ sqrt(p[1]^2 + p[2]^2)
end

function getKernelParams(pK::T) where {T <: LM_AHOConstantKernelParameters}
    return pK.H
end




"""
Kernel defined by a real number ``θ``, such that the kernel is a phase rotation
```math
K = cos(θ) + isin(θ)
```
To get H we use the frommula for the square root
```math
H = cos(θ/2) + isin(θ/2)
```
"""
struct LM_AHO_ConstKernel_θ <: LM_AHOConstantKernelParameters
    model::LM_AHO
    θ
    function LM_AHO_ConstKernel_θ(model::LM_AHO; θs=[0.])
        new(model,θs)
    end
end

function getKernelParams(pK::LM_AHO_ConstKernel_θ)
    return pK.θ
end


function K(u,p,::LM_AHO_ConstKernel_θ)
    return [cos(p[1]),sin(p[1])]
end


function sqrtK(u,p,pK::LM_AHO_ConstKernel_θ)
    return [cos(p[1]*0.5),sin(p[1]*0.5)]
end




######################
abstract type U1ConstantKernelParmeters <: ConstantKernelParameters end


"""
Kernel defined by a real number ``θ``, such that the kernel is a phase rotation
```math
K = cos(θ) + isin(θ)
```
To get H we use the frommula for the square root
```math
H = cos(θ/2) + isin(θ/2)
```
"""
struct U1_ConstKernel <: ConstantKernelParameters
    model::U1
    θ
    function U1_ConstKernel(model::U1; θs=[0.])
        new(model,θs)
    end
end

function getKernelParams(pK::U1_ConstKernel)
    return pK.θs
end

function getKernelParamsSim(pK::U1_ConstKernel)
    return pK.θs
end

function updateKernel!(::U1_ConstKernel)
    nothing
end

function K(u,p,::U1_ConstKernel)
    return [cos(p[1]),sin(p[1])]
end

function dK(u,p,::U1_ConstKernel)
    return nothing
end

function sqrtK(u,p,::U1_ConstKernel)
    return [cos(p[1]*0.5),sin(p[1]*0.5)]
end

function K_dK(u,p,pK::U1_ConstKernel)
    return K(u,p,pK),dK(u,p,pK)
end





"""
Structure to store the kernel functions
"""
struct ConstantKernel{pType<:KernelParameters, KType <: Function, dKType <: Function,K_dKType <: Function, sqrtKType <: Function} <: Kernel
    pK::pType
    K::KType
    dK::dKType
    K_dK::K_dKType
    sqrtK::sqrtKType

    function ConstantKernel(pK::T) where {T<:ConstantKernelParameters}        
        _K(u,p) = K(u,p,pK) 
        _dK(u,p) = dK(u,p,pK) 
        _K_dK(u,p) = K_dK(u,p,pK)
        _sqrtK(u,p) = sqrtK(u,p,pK)
        new{T,typeof(_K),typeof(_dK),typeof(_K_dK),typeof(_sqrtK)}(pK,_K,_dK,_K_dK,_sqrtK) 
    end
end

function ConstantKernel(M::AHO; kernelType=:expiP)
    if kernelType == :K
        pK = AHO_ConstKernel_K(M)  
    elseif kernelType == :H
        pK = AHO_ConstKernel_expH(M)
    elseif kernelType == :sincos
        pK = AHO_ConstKernel_sincos(M)    
    elseif kernelType == :expiP
        pK = AHO_ConstKernel_expiP(M)
    elseif kernelType == :expA
        pK = AHO_ConstKernel_expA(M) 
    elseif kernelType == :HexpA
        pK = AHO_ConstKernel_HexpA(M)   
    elseif kernelType == :expiHerm
        pK = AHO_ConstKernel_expiHerm(M)   
    elseif kernelType == :expiSym
        pK = AHO_ConstKernel_expiHerm(M)   
    elseif kernelType == :inM_expiP
        pK = AHO_ConstKernel_invM_expiP(M)  
    end

    return ConstantKernel(pK)
end



function ConstantFreeKernel(M::AHO; m=M.m, g=1)
    @unpack a,t_steps,κ = M.contour 
    A = zeros(Complex{typeof(m)},t_steps,t_steps)
    #Abuf = Zygote.Buffer(A)
    #Abuf .= zeros(eltype(A),t_steps,t_steps)
    for j in 1:t_steps
        jm1 = mod1(j-1,t_steps)
        jp1 = mod1(j+1,t_steps)
        A[j,j] = g*((1/a[jm1]) + (1/a[j])) - 0.5*(a[jm1] + a[j])*m
        A[j,jp1] = -g*(1/a[j])
        A[j,jm1] = -g*(1/a[jm1])
    end
    #A = copy(Abuf)
    A = -im*A
    @show ishermitian(A)
    @show issymmetric(A)
    K = inv(A)
    #KR = hcat(vcat(real(K),imag(K)),vcat(-imag(K),real(K)))

    #schurK = schur(K)
    #sqrtK = schurK.vectors * sqrt(UpperTriangular(schurK.T)) * schurK.vectors'
    #vals,vecs = eigen(K)
    
    
    #=sqrtK = ff(KR)#vecs * Diagonal(sqrt.(vals .+ 0 * im)) * inv(vecs)
    @show ishermitian(K)
    @show issymmetric(K)

    @show ishermitian(sqrtK)
    @show issymmetric(sqrtK)

    sqrtKR = reshape(sqrtK,2t_steps,2t_steps)[:,1:t_steps]#vcat(real(sqrtK),imag(sqrtK))
    #return ConstantKernel(AHO_ConstKernel(sqrtKR))=#
    
    #return ConstantKernel(AHO_ConstKernel_K([real(K) ; imag(K)]))
    
    AiP = log(K)
    return ConstantKernel(AHO_ConstKernel_expA([real(AiP) ; imag(AiP)]))
    #return sqrtKR
end






function ConstantKernel(M::LM_AHO)
    pK = LM_AHO_ConstKernel(M)
    return ConstantKernel(pK)
end

function ConstantKernel(M::U1)
    pK = U1_ConstKernel(M)
    return ConstantKernel(pK)
end

function getKernelParams(kernel::ConstantKernel{pKType}) where {pKType <: ConstantKernelParameters}
    return getKernelParams(kernel.pK)
end

function getKernelParamsSim(kernel::ConstantKernel{pKType}) where {pKType <: ConstantKernelParameters}
    return getKernelParamsSim(kernel.pK)
end

function setKernel!(kernel::ConstantKernel{pKType},v) where {pKType <: ConstantKernelParameters}
    kernel.pK.H .= v
end

function setKernel!(kernel::ConstantKernel{LM_AHO_ConstKernel_θ},v)
    kernel.pK.θs .= v
end


###############

abstract type AHO_FieldKernelParameters <: KernelParameters end
abstract type LM_FieldKernelParameters <: KernelParameters end

struct LM_AHO_FieldKernel <: LM_FieldKernelParameters
    model::LM_AHO
    θ::Float64
    function LM_AHO_FieldKernel(model::LM_AHO,θ::Float64)
        new(model,θ)
    end
end

function getKernelParams(pK::LM_AHO_FieldKernel)
    @unpack θ = pK
    return [θ]
end

function getKernelParamsSim(pK::LM_AHO_FieldKernel)
    @unpack θ = pK
    return [θ]
end

function updateKernel!(pK::LM_AHO_FieldKernel)
    nothing
end

function f(u,pK::LM_AHO_FieldKernel)
    @unpack σ,λ = pK.model
    
    _σ = σ[1] + im*σ[2]
    return exp.((u[1]^2 - u[2]^2) .* ((λ/6)/_σ)) .* (cos(2*u[1]*u[2]*((λ/6)/_σ)) + im*sin(2*u[1]*u[2]*((λ/6)/_σ)))
end

function K(u,p,pK::LM_AHO_FieldKernel; _f = f(u,pK))
    @unpack σ,λ = pK.model
    θ, = p
    
    _σ = σ[1] + im*σ[2]
    _K = (1/abs.(_σ))*_f*(cos(θ) - im*sin(θ)) + (6/abs(λ))*(1 .- _f)
    return real(_K),imag(_K)
end

function dK(u,p,pK::LM_AHO_FieldKernel;_f = f(u,pK))
    @unpack σ,λ = pK.model
    θ, = p

    _σ = σ[1] + im*σ[2]
    _dK = 2*(u[1] + im*u[2])*((λ/6)/_σ)*((1/abs.(_σ))*(cos(θ) - im*sin(θ)) - (6/abs(λ)))*_f
    return real(_dK),imag(_dK)
end

function K_dK(u,p,pK::LM_AHO_FieldKernel)
    _f = f(u,pK)
    return K(u,p,pK;_f=_f),dK(u,p,pK;_f=_f)
end

function sqrtK(u,p,pK::LM_AHO_FieldKernel)
    _KRe, _KIm = K(u,p,pK) 
    sqrtK = sqrt(_KRe + im*_KIm)
    return real(sqrtK),imag(sqrtK)
end





##### LM_AHO NNKernel

struct LM_AHO_NNKernel <: LM_FieldKernelParameters
    model::LM_AHO
    ann
    re
    function LM_AHO_NNKernel(model::LM_AHO,NN)
        p,re = Flux.destructure(NN)
        p = convert(Vector{Float64},p)
        new(model,p,re)
    end
end

function getKernelParams(pK::LM_AHO_NNKernel)
    @unpack ann = pK
    return ann
end

function getKernelParamsSim(pK::LM_AHO_NNKernel)
    @unpack ann = pK
    return ann
end

function updateKernel!(pK::LM_AHO_NNKernel)
    nothing
end

function K(u,p,pK::LM_AHO_NNKernel)
    #sqrtK=pK.re(p)([u[1] + im*u[2]])[1]
    #return real(sqrtK)^2 - imag(sqrtK)^2, 2*real(sqrtK)*imag(sqrtK) 

    sqrtK=pK.re(p)(u)
    sqrtK = sqrtK / sqrt(sqrtK[1]^2 + sqrtK[2]^2)
    return sqrtK[1]^2 - sqrtK[2]^2, 2*sqrtK[1]*sqrtK[2] 
end

function dK(u,p,pK::LM_AHO_NNKernel)
    dKs = ForwardDiff.gradient(x -> K(x,p,pK)[1], u)
    #dKs = Zygote.gradient(x -> K(x,p,pK)[1], u)[1]

    # derivative of holomorphic function is conjugate of gradient of the real part
    dKs = [dKs[1],-dKs[2]] 
    return dKs
end

function sqrtK(u,p,pK::LM_AHO_NNKernel)
    #_sqrtK = pK.re(p)([u[1] + im*u[2]])
    #return real(_sqrtK[1]), imag(_sqrtK[1])

    _sqrtK = pK.re(p)(u)
    _sqrtK = _sqrtK / sqrt(_sqrtK[1]^2 + _sqrtK[2]^2)
    return _sqrtK[1], _sqrtK[2]
end

function K_dK(u,p,pK::LM_AHO_NNKernel)
    return K(u,p,pK),dK(u,p,pK)
end

##### LM_AHO NNKernel

struct LM_AHO_NNKernel1 <: LM_FieldKernelParameters
    model::LM_AHO
    ann
    re
    function LM_AHO_NNKernel1(model::LM_AHO,NN)
        p,re = Flux.destructure(NN)
        p = convert(Vector{Float64},p)
        new(model,p,re)
    end
end

function getKernelParams(pK::LM_AHO_NNKernel1)
    @unpack ann = pK
    return ann
end

function getKernelParamsSim(pK::LM_AHO_NNKernel1)
    @unpack ann = pK
    return ann
end

function updateKernel!(pK::LM_AHO_NNKernel1)
    nothing
end

function K(u,p,pK::LM_AHO_NNKernel1)
    NN=pK.re(p)(u)
    #return sum(NN[1:2:end] .* cos.(NN[2:2:end])),sum(NN[1:2:end] .* sin.(NN[2:2:end]))
    return sum(cos.(NN[1:end])),sum(sin.(NN[1:end]))
end

function dK(u,p,pK::LM_AHO_NNKernel1)
    dKs = ForwardDiff.gradient(x -> K(x,p,pK)[1], u)

    # derivative of holomorphic function is conjugate of gradient of the real part
    dKs = [dKs[1],-dKs[2]] 
    return dKs
end

function sqrtK(u,p,pK::LM_AHO_NNKernel1)
    _K = K(u,p,pK)
    sqrtK = sqrt(_K[1] + im*_K[2])
    return real(sqrtK),imag(sqrtK)
end

function K_dK(u,p,pK::LM_AHO_NNKernel1)
    return K(u,p,pK),dK(u,p,pK)
end

##### U1 NNKernel

struct U1_NNKernel <: LM_FieldKernelParameters
    model::U1
    ann
    re
    function U1_NNKernel(model::U1,NN)
        p,re = Flux.destructure(NN)
        p = convert(Vector{Float64},p)
        new(model,p,re)
    end
end

function getKernelParams(pK::U1_NNKernel)
    @unpack ann = pK
    return ann
end

function getKernelParamsSim(pK::U1_NNKernel)
    @unpack ann = pK
    return ann
end

function updateKernel!(pK::U1_NNKernel)
    nothing
end

function K(u,p,pK::U1_NNKernel)
    NN=pK.re(p)(u)
    #return sum(NN[1:2:end] .* cos.(NN[2:2:end])),sum(NN[1:2:end] .* sin.(NN[2:2:end]))
    return sum(cos.(NN[1:end])),sum(sin.(NN[1:end]))
end

function dK(u,p,pK::U1_NNKernel)
    dKs = ForwardDiff.gradient(x -> K(x,p,pK)[1], u)

    # derivative of holomorphic function is conjugate of gradient of the real part
    dKs = [dKs[1],-dKs[2]] 
    return dKs
end

function sqrtK(u,p,pK::U1_NNKernel)
    _K = K(u,p,pK)
    sqrtK = sqrt(_K[1] + im*_K[2])
    return real(sqrtK),imag(sqrtK)
end

function K_dK(u,p,pK::U1_NNKernel)
    return K(u,p,pK),dK(u,p,pK)
end

##### U1 NNKernel1

struct U1_NNKernel1 <: LM_FieldKernelParameters
    model::U1
    ann
    re
    function U1_NNKernel1(model::U1,NN)
        p,re = Flux.destructure(NN)
        p = convert(Vector{Float64},p)
        new(model,p,re)
    end
end

function getKernelParams(pK::U1_NNKernel1)
    @unpack ann = pK
    return ann
end

function getKernelParamsSim(pK::U1_NNKernel1)
    @unpack ann = pK
    return ann
end

function updateKernel!(pK::U1_NNKernel1)
    nothing
end

function K(u,p,pK::U1_NNKernel1)
    _sqrtK = sqrtK(u,p,pK)

    return [_sqrtK[1]^2 - _sqrtK[2]^2, 2*_sqrtK[1]*_sqrtK[2]]
end

function dK(u,p,pK::U1_NNKernel1)
    dKs = ForwardDiff.gradient(x -> K(x,p,pK)[1], u)

    # derivative of holomorphic function is conjugate of gradient of the real part
    dKs = [dKs[1],-dKs[2]] 
    return dKs
end

function sqrtK(u,p,pK::U1_NNKernel1)
    #H = 2*sin(u[1] + im*u[2])^2
    #return real(H),imag(H)
    return pK.re(p)(u)
end

function K_dK(u,p,pK::U1_NNKernel1)
    return K(u,p,pK),dK(u,p,pK)
end

##### AHO FieldKernel

struct AHO_FieldKernel <: AHO_FieldKernelParameters
    model::AHO
    p::Matrix{Float64}
end

function getKernelParams(pK::AHO_FieldKernel)
    return pK.p
end

function getKernelParamsSim(pK::AHO_FieldKernel)
    return pK.p
end

function f(u,pK::AHO_FieldKernel)
    
    u2 = mean(u[1:div(end,2)].^2 .- u[div(end,2)+1:end].^2  .+ 2 * im* u[1:div(end,2)] .* u[div(end,2)+1:end])
    return exp(-u2*24)
end

function K(u,p,pK::AHO_FieldKernel;_f=f(u,pK))
    _K = _f*exp(im*p[1:div(end,2),:]) + (1-_f)*exp(im*p[div(end,2)+1:end,:])
    return real(_K),imag(_K)
end

function dK(u,p,pK::AHO_FieldKernel;_f=f(u,pK))
    
    _u = u[1:div(end,2)] .+ im*u[div(end,2)+1:end]
    dK = -(2*24)*_f*(exp(im*p[1:div(end,2),:]) - exp(im*p[div(end,2)+1:end,:]))*_u

    return [real(dK),imag(dK)]
end

function sqrtK(u,p,pK::AHO_FieldKernel;_f=f(u,pK))
    _K = _f*exp(im*p[1:div(end,2),:]) + (1-_f)*exp(im*p[div(end,2)+1:end,:])
    _sqrtK = sqrt(_K)
    return [real(_sqrtK);imag(_sqrtK)]
end

function K_dK(u,p,pK::AHO_FieldKernel)
    _f = f(u,pK)
    return K(u,p,pK;_f=_f),dK(u,p,pK;_f=_f)
end

##### AHO NNKernel

struct AHO_NNKernel <: AHO_FieldKernelParameters
    model::AHO
    ann
    re
    function LM_AHO_NNKernel(model::LM_AHO,NN)
        p,re = Flux.destructure(NN)
        p = convert(Vector{Float64},p)
        new(model,p,re)
    end
end

function getKernelParams(pK::AHO_NNKernel)
    @unpack ann = pK
    return ann
end

function K(u,p,pK::AHO_NNKernel)
    #sqrtK=pK.re(p)([u[1] + im*u[2]])[1]
    #return real(sqrtK)^2 - imag(sqrtK)^2, 2*real(sqrtK)*imag(sqrtK) 

    sqrtK=pK.re(p)(u)
    K = sqrtK^2
    return [real(K);imag(K)] 
end

function dK(u,p,pK::AHO_NNKernel)
    dKs = ForwardDiff.gradient(x -> K(x,p,pK)[1], u)
    #dKs = Zygote.gradient(x -> K(x,p,pK)[1], u)[1]

    # derivative of holomorphic function is conjugate of gradient of the real part
    dKs = [dKs[1],-dKs[2]] 
    return dKs
end

function sqrtK(u,p,pK::AHO_NNKernel)
    #_sqrtK = pK.re(p)([u[1] + im*u[2]])
    #return real(_sqrtK[1]), imag(_sqrtK[1])

    _sqrtK = pK.re(p)(u)
    _sqrtK = _sqrtK / sqrt(_sqrtK[1]^2 + _sqrtK[2]^2)
    return _sqrtK[1], _sqrtK[2]
end

function K_dK(u,p,pK::AHO_NNKernel)
    return K(u,p,pK),dK(u,p,pK)
end




abstract type FieldDependentKernel  <: Kernel end

struct FunctionalKernel{pType<:KernelParameters, KType <: Function, dKType <: Function,K_dKType <: Function, sqrtKType <: Function} <: FieldDependentKernel
    pK::pType
    K::KType
    dK::dKType
    K_dK::K_dKType
    sqrtK::sqrtKType

    function FunctionalKernel(pK::T) where {T<:KernelParameters}        
        _K(u,p) = K(u,p,pK) 
        _dK(u,p) = dK(u,p,pK) 
        _K_dK(u,p) = K_dK(u,p,pK)
        _sqrtK(u,p) = sqrtK(u,p,pK)
        new{T,typeof(_K),typeof(_dK),typeof(_K_dK),typeof(_sqrtK)}(pK,_K,_dK,_K_dK,_sqrtK) 
    end
end


function FieldDependentKernel(p::LM_AHO; θ=pi/2)
    pK = LM_AHO_FieldKernel(p,θ)
    return FunctionalKernel(pK)
end

function FieldDependentKernel(M::AHO)
    @unpack t_steps = M.contour 
    p = zeros(2*t_steps,t_steps)
    pK = AHO_FieldKernel(M,p)
    return FunctionalKernel(pK)
end

function NNDependentKernel(model::LM_AHO;N=8)

    #ann = Chain(Dense(2,N),Dense(N,N),Dense(N,2))
    ann = Chain(Dense(2,N,tanh),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(1,N),Dense(N,N),Dense(N,N),Dense(0.01*randn(1, N),[1.]))
    #ann = Chain(Dense(2,N,leakyrelu),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(zeros(2, 2),[1.,0.]))

    pK = LM_AHO_NNKernel(model,ann)
    return FunctionalKernel(pK)
end

function NNDependentKernel1(model::LM_AHO;N=8)

    #ann = Chain(Dense(2,N),Dense(N,N),Dense(N,2))
    #ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(4, N),[1., 0.,0.,0.]))
    ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(1, N),[0.]))
    #ann = Chain(Dense(1,N),Dense(N,N),Dense(N,N),Dense(0.01*randn(1, N),[1.]))
    #ann = Chain(Dense(2,N,leakyrelu),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(zeros(2, 2),[1.,0.]))

    pK = LM_AHO_NNKernel1(model,ann)
    return FunctionalKernel(pK)
end

function NNDependentKernel(model::U1;N=8)

    #ann = Chain(Dense(2,N),Dense(N,N),Dense(N,2))
    #ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(4, N),[1., 0.,0.,0.]))
    ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(1, N),[0.]))
    #ann = Chain(Dense(1,N),Dense(N,N),Dense(N,N),Dense(0.01*randn(1, N),[1.]))
    #ann = Chain(Dense(2,N,leakyrelu),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(zeros(2, 2),[1.,0.]))

    pK = U1_NNKernel(model,ann)
    
    return FunctionalKernel(pK)
end

function NNDependentKernel1(model::U1;N=8)

    #ann = Chain(Dense(2,N),Dense(N,N),Dense(N,2))
    #ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(4, N),[1., 0.,0.,0.]))
    ann = Chain(Dense(2,N,tanh),Dense(N,N,tanh),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(1,N),Dense(N,N),Dense(N,N),Dense(0.01*randn(1, N),[1.]))
    #ann = Chain(Dense(2,N,leakyrelu),Dense(0.001*randn(2, N),[1., 0.]))
    #ann = Chain(Dense(zeros(2, 2),[1.,0.]))

    pK = U1_NNKernel1(model,ann)
    
    return FunctionalKernel(pK)
end


function getKernelParams(kernel::FieldDependentKernel)
    return getKernelParams(kernel.pK)
end

function getKernelParamsSim(kernel::FieldDependentKernel)
    return getKernelParamsSim(kernel.pK)
end

