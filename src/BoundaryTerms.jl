
export getBoundaryTerms, calcBoundaryTerms, calcBoundaryTermsCorrections

#######################################
abstract type AbstractBoundaryTerms end

struct BoundaryTerms{MType,YT,LType,L2Type} <: AbstractBoundaryTerms
    model::MType
    Ys::Vector{YT}
    Xs::Union{Vector{YT},Nothing}
    L_CO::LType
    L2_CO::L2Type
    function BoundaryTerms(model::MType,Ys::Vector{YT},L_CO::LType) where {MType <: Model,YT,LType}
        new{MType,YT,LType,Nothing}(model,Ys,nothing,L_CO,nothing)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},Xs::Union{Vector{YT},Nothing},L_CO::LType) where {MType <: Model,YT,LType}
        new{MType,YT,LType,Nothing}(model,Ys,Xs,L_CO,nothing)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},L_CO::LType,L2_CO::L2Type) where {MType <: Model,YT,LType,L2Type}
        new{MType,YT,LType,L2Type}(model,Ys,nothing,L_CO,L2_CO)
    end
    function BoundaryTerms(model::MType,Ys::Vector{YT},Xs::Union{Vector{YT},Nothing},L_CO::LType,L2_CO::L2Type) where {MType <: Model,YT,LType,L2Type}
        new{MType,YT,LType,L2Type}(model,Ys,Xs,L_CO,L2_CO)
    end
end


function Lcx(u,p,σ,λ,::ConstantKernel{LM_AHOConstantKernelParameters})

    _A = similar(u)

    _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
    _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
    
    KRe = p[1]^2 - p[2]^2
    KIm = 2*p[1]*p[2]

    return [KRe*_A[1] - KIm*_A[2],
            KRe*_A[2] + KIm*_A[1]]
end

"""
    Second order boundary term for x
        L2x = K^2((σ^2 + λ)x + 2σλ/3 x^3 + λ^2/12 x^5)
"""
function L2cx(u,p,σ,λ,::ConstantKernel{LM_AHOConstantKernelParameters})

    _u2 = similar(u)
    _u2[1] = (u[1]*u[1] - u[2]*u[2])
    _u2[2] = 2u[1]*u[2]

    _u3 = similar(u)
    _u3[1] = (u[1]*_u2[1] - u[2]*_u2[2])
    _u3[2] = u[1]*_u2[2] + _u2[1]*u[2]

    _u5 = similar(u)
    _u5[1] = (_u3[1]*_u2[1] - _u3[2]*_u2[2])
    _u5[2] = _u3[1]*_u2[2] + _u2[1]*_u3[2]

    _A = similar(u)
    _A[1] = -( (σ[1] + λ) * u[1] - σ[2] * u[2] 
             + (2/3)* λ * (σ[1] * _u3[1] - σ[2] * _u3[2]) 
             + (λ^2/12) * _u5[1] )
    _A[2] = -( (σ[1] + λ) * u[2] + σ[2] * u[1] 
             + (2/3)* λ * (σ[2] * _u3[1] + σ[1] * _u3[2]) 
             + (λ^2/12) * _u5[2] )
    
    KRe = p[1]^2 - p[2]^2
    KIm = 2*p[1]*p[2]

    K2Re = KRe^2 - KIm^2
    K2Im = 2KRe*KIm

    return [K2Re*_A[1] - K2Im*_A[2],
            K2Re*_A[2] + K2Im*_A[1]]
end

function Lcx2(u,p,σ,λ,::ConstantKernel{LM_AHOConstantKernelParameters})

        _A = similar(u)
        _u2 = similar(u)
    
        _u2[1] = u[1]^2 - u[2]^2
        _u2[2] = 2*u[1]*u[2]
    
        _A[1] = -(σ[1] * _u2[1] - σ[2] * _u2[2] + (1/6)*λ * (_u2[1]^2 - _u2[2]^2))
        _A[2] = -(σ[1] * _u2[2] + σ[2] * _u2[1] + (1/6)*λ * (2*_u2[1]*_u2[2]))
        
        KRe = p[1]^2 - p[2]^2
        KIm = 2*p[1]*p[2]
    
        return 2. * [KRe + (KRe*_A[1] - KIm*_A[2]),
                     KIm + (KRe*_A[2] + KIm*_A[1])]
end

"""
    Second order boundary term for x^2
        L2x^2 = K^2((σ^2 + λ)x + 2σλ/3 x^3 + λ^2/12 x^5)
"""
function L2cx2(u,p,σ,λ,::ConstantKernel{LM_AHOConstantKernelParameters})

    _A = similar(u)
    _u2 = similar(u)
    _u4 = similar(u)
    _u6 = similar(u)

    _u2[1] = u[1]^2 - u[2]^2
    _u2[2] = 2*u[1]*u[2]

    _u4[1] = _u2[1]^2 - _u2[2]^2
    _u4[2] = 2*_u2[1]*_u2[2]

    _u6[1] = _u2[1]*_u4[1] - _u2[2]*_u4[2]
    _u6[2] = _u2[1]*_u4[2] + _u2[2]*_u4[1]

    σ2Re = σ[1]^2 - σ[2]^2
    σ2Im = 2*σ[1]*σ[2]

    _A[1] = - (2*σ[1] + 2*(λ - σ2Re) * _u2[1] - 2*( - σ2Im) * _u2[2] 
                - λ * (σ[1] * _u4[1] - σ[2]* _u4[2]) - (1/9)* λ^2 * _u6[1])
    _A[2] = - (2*σ[2] + 2*(λ - σ2Re) * _u2[2] + 2*( - σ2Im) * _u2[1] 
                - λ * (σ[1] * _u4[2] + σ[2]* _u4[1]) - (1/9)* λ^2 * _u6[2])
    
    KRe = p[1]^2 - p[2]^2
    KIm = 2*p[1]*p[2]

    K2Re = KRe^2 - KIm^2
    K2Im = 2KRe*KIm

    return 2. * [K2Re*_A[1] - K2Im*_A[2],
                 K2Re*_A[2] + K2Im*_A[1]]
end



#=function getBoundaryTermsObservables(KP::KernelProblem{LM_AHO})

    @unpack model, kernel = KP
    @unpack σ, λ = model
    p = getKernelParams(kernel)

    LcO(x) = [Lcx2(x,p,σ,λ,kernel)..., Lcx(x,p,σ,λ,kernel)...]
    L2cO(x) = [L2cx2(x,p,σ,λ,kernel)..., L2cx(x,p,σ,λ,kernel)...]

    return LcO, L2cO
end=#





function Lc_x_x2(u,p,model::LM_AHO,kernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack σ, λ = model

    _A = similar(u)

    _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
    _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
    
    (KRe,KIm),dK = K_dK(u,p)


    _AxRe = (KRe*_A[1] - KIm*_A[2])*u[1] - (KRe*_A[2] + KIm*_A[1])*u[2]
    _AxIm = (KRe*_A[1] - KIm*_A[2])*u[2] + (KRe*_A[2] + KIm*_A[1])*u[1]

    if isnothing(dK)
        return [KRe*_A[1] - KIm*_A[2] 
                KRe*_A[2] + KIm*_A[1]
             2*(KRe + _AxRe)
             2*(KIm + _AxIm)]
    else
        dKRe, dKIm = dK
        dKRex = dKRe*u[1] - dKIm*u[2]
        dKImx = dKRe*u[2] + dKIm*u[1]
        return [KRe*_A[1] - KIm*_A[2] + dK[1]
                KRe*_A[2] + KIm*_A[1] + dK[2]
             2*(dKRex + KRe + _AxRe)
             2*(dKImx + KIm + _AxIm)]
    end
end

function L2c_x_x2(u,p,model::LM_AHO,kernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack σ, λ = model

    _A = similar(u)

    _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
    _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
    
    (KRe,KIm),dK = K_dK(u,p)

    _AxRe = (KRe*_A[1] - KIm*_A[2])*u[1] - (KRe*_A[2] + KIm*_A[1])*u[2]
    _AxIm = (KRe*_A[1] - KIm*_A[2])*u[2] + (KRe*_A[2] + KIm*_A[1])*u[1]

    if isnothing(dK)
        return [KRe*_A[1] - KIm*_A[2] 
                KRe*_A[2] + KIm*_A[1]
             2*(KRe + _AxRe)
             2*(KIm + _AxIm)]
    else
        dKRe, dKIm = dK
        dKRex = dKRe*u[1] - dKIm*u[2]
        dKImx = dKRe*u[2] + dKIm*u[1]
        return [KRe*_A[1] - KIm*_A[2] + dK[1]
                KRe*_A[2] + KIm*_A[1] + dK[2]
             2*(dKRex + KRe + _AxRe)
             2*(dKImx + KIm + _AxIm)]
    end
end




function getBoundaryTermsObservables(KP::KernelProblem{LM_AHO}; p = getKernelParams(KP.kernel))

    @unpack model, kernel = KP
    #@unpack σ, λ = model

    LcO(x) = [Lc_x_x2(x,p,model,kernel)...]#[Lcx2(x,p,σ,λ,kernel)..., Lcx(x,p,σ,λ,kernel)...]
    #L2cO(x) = [L2cx2(x,p,σ,λ,kernel)..., L2cx(x,p,σ,λ,kernel)...]

    return LcO, nothing
end







function L_CO_U1(x,y,k,p)
    z = x + im*y
    return im*k*(im*k + im*p.β*sin(z)-p.s*z)*exp(im*k*z)
end

function L2_CO_U1(x,y,k,p)
    s = p.s
    β = p.β
    z = x + im*y
    return k*exp(im*k*z)*(
         k^3 + 2k*s + 2*im*k^2*s*z + im*s^2*z - k*s^2*z^2 + (1 + 2*k^2 + s + 2*im*k*s*z)*β*sin(z)
       + k*β^2*sin(z)^2 + β*cos(z)*(-2*im*k + s*z - im*β*sin(z))
    )    
end


function Lc_expi1_expi2(u,p,model::U1,kernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack β, s = model

    x,y = u
    z = x + im*y

    (KRe,KIm),dK = K_dK(u,p)
    _K = KRe + im*KIm


    
    
    if isnothing(dK)
        Lc_expi1 = im*1*_K*(im*1 + im*β*sin(z)-s*z)*exp(im*1*z)
        Lc_expi2 = im*2*_K*(im*2 + im*β*sin(z)-s*z)*exp(im*2*z)

        return [real(Lc_expi1)
                imag(Lc_expi1)
                real(Lc_expi2)
                imag(Lc_expi2)]

    else
        _dK = dK[1] + im*dK[2]
        Lc_expi1 = im*1*_K*(im*1 + im*β*sin(z)-s*z)*exp(im*1*z) - im*1*_dK*exp(im*1*z)
        Lc_expi2 = im*2*_K*(im*2 + im*β*sin(z)-s*z)*exp(im*2*z) - im*2*_dK*exp(im*2*z)

        return [real(Lc_expi1)
                imag(Lc_expi1)
                real(Lc_expi2)
                imag(Lc_expi2)]
    end
end

function L2c_expi1_expi2(u,p,model::U1,kernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack σ, λ = model

    _A = similar(u)

    _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
    _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
    
    (KRe,KIm),dK = K_dK(u,p)

    _AxRe = (KRe*_A[1] - KIm*_A[2])*u[1] - (KRe*_A[2] + KIm*_A[1])*u[2]
    _AxIm = (KRe*_A[1] - KIm*_A[2])*u[2] + (KRe*_A[2] + KIm*_A[1])*u[1]

    if isnothing(dK)
        return [KRe*_A[1] - KIm*_A[2] 
                KRe*_A[2] + KIm*_A[1]
             2*(KRe + _AxRe)
             2*(KIm + _AxIm)]
    else
        dKRe, dKIm = dK
        dKRex = dKRe*u[1] - dKIm*u[2]
        dKImx = dKRe*u[2] + dKIm*u[1]
        return [KRe*_A[1] - KIm*_A[2] + dK[1]
                KRe*_A[2] + KIm*_A[1] + dK[2]
             2*(dKRex + KRe + _AxRe)
             2*(dKImx + KIm + _AxIm)]
    end
end



function getBoundaryTermsObservables(KP::KernelProblem{U1}; p = getKernelParams(KP.kernel))

    @unpack model, kernel = KP

    LcO(x) = [Lc_expi1_expi2(x,p,model,kernel)...]

    return LcO, nothing
end










function Lc_x_x2_x0xt(u,p,m,λ,a,a_m1,gp1,gm1,κ,::ConstantKernel{AHOConstantKernelParameters})

    Lcx = similar(u)
    Lcx2 = similar(u)
    Lcx0xt = similar(u)

    xR =  @view u[1:div(end,2)]
    xI =  @view u[div(end,2)+1:end]

    pre_fac = (1 / abs(a[1]))
    
    ARe = - pre_fac .* 0.5 .* (
        2. .* ( real.(a_m1) .* (xI .- xI[gp1])  .- imag.(a_m1 .- κ).*(xR .- xR[gp1]) ) ./ abs.(a_m1).^2
    .- 2. .* ( real.(a)  .* (xI[gm1] .- xI) .- imag.(a .- κ)  .* (xR[gm1] .- xR) )./ abs.(a).^2
    .- real.(a_m1 .+ a) .* (m .* xI .+ (1/6)*λ .* (-(xI.^3) .+ 3xI.*(xR.^2))) 
    .- imag.(a_m1 .+ a) .* (m .* xR .+ (1/6)*λ .* (xR.^3 .- 3xR.*(xI.^2))) 
    )     
    
    AIm = pre_fac .* 0.5 .* (
        2. .* (real.(a_m1).*(xR .- xR[gp1]) .+ imag.(a_m1 .- κ).*(xI .- xI[gp1])) ./ abs.(a_m1).^2
    .- 2. .* (real.(a).*(xR[gm1] .- xR)   .+ imag.(a .- κ).*(xI[gm1] .- xI)) ./ abs.(a).^2
    .- real.(a_m1 .+ a) .* (m.*xR .+ (1/6)*λ .* ((xR.^3) .- 3xR.*(xI.^2))) 
    .+ imag.(a_m1 .+ a) .* (m.*xI .+ (1/6)*λ .* (-xI.^3 .+ 3xI.*(xR.^2))) 
    )

    kReT = (p[1:div(end,2),:]*transpose(p[1:div(end,2),:]) 
                - p[div(end,2)+1:end,:]*transpose(p[div(end,2)+1:end,:]))
    kImT = (p[1:div(end,2),:]*transpose(p[div(end,2)+1:end,:]) 
                + p[div(end,2)+1:end,:]*transpose(p[1:div(end,2),:]))
    
    Lcx[1:div(end,2)]     .= -vec((transpose(ARe)*kReT .- transpose(AIm)*kImT))
    Lcx[div(end,2)+1:end] .= -vec((transpose(AIm)*kReT .+ transpose(ARe)*kImT))
    

    SjKTRe = vec(transpose(ARe)*kReT .- transpose(AIm)*kImT)
    SjKTIm = vec(transpose(AIm)*kReT .+ transpose(ARe)*kImT)
    Lcx2[1:div(end,2)]     .= 2*diag(kReT) .- 2*(SjKTRe .* xR .- SjKTIm .* xI)
    Lcx2[div(end,2)+1:end] .= 2*diag(kImT) .- 2*(SjKTRe .* xI .+ SjKTIm .* xR)
    
    
    SjKj0Re = ARe.*view(kReT,:,1) .- AIm.*view(kImT,:,1)
    SjKj0Im = AIm.*view(kReT,:,1) .+ ARe.*view(kImT,:,1)
    Lcx0xt[1:div(end,2)]     .= vec(kReT[1,:] .+ kReT[:,1]) .- (SjKTRe .* xR[1] .- SjKTIm .* xI[1]) .- (SjKj0Re .* xR .- SjKj0Im .* xI)
    Lcx0xt[div(end,2)+1:end] .= vec(kImT[1,:] .+ kImT[:,1]) .- (SjKTRe .* xI[1] .+ SjKTIm .* xR[1]) .- (SjKj0Re .* xI .+ SjKj0Im .* xR)
    return [Lcx...,Lcx2...,Lcx0xt...]
end






function getBoundaryTermsObservables(KP::KernelProblem{AHO})

    @unpack model, kernel = KP
    @unpack m, λ, contour = model
    @unpack a, t_steps = contour
    p = getKernelParams(kernel)

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]

    κ = 0. * im

    LcO(x) = Lc_x_x2_x0xt(x,p,m,λ,a,a_m1,gp1,gm1,κ,kernel)
    #LcO(x) = [Lcx(x,p,m,λ,a,a_m1,gp1,gm1,κ,kernel)..., Lcx2(x,p,m,λ,a,a_m1,gp1,gm1,κ,kernel)...]
    #L2cO(x) = [L2cx2(x,p,m,λ,kernel)..., L2cx(x,p,m,λ,kernel)...]

    return LcO, nothing#L2cO
end







function getBoundaryTerms(KP::KernelProblem;Ys=nothing,Xs=nothing,kwargs...)

    @unpack model, kernel = KP

    if isnothing(Ys)
        Ys = collect(0:0.1:8)
    end

    L_CO,L2_CO = getBoundaryTermsObservables(KP;kwargs...)
    return BoundaryTerms(model,Ys,Xs,L_CO,L2_CO)
end






function calcBoundaryTerms(sol,BT::BoundaryTerms{MType,YT,L,L2};T=Float64, witherror=true) where {MType <: Model,YT,L,L2}
    @unpack Ys, Xs = BT
    NTr = length(sol)

    realDir = !isnothing(Xs)
    d = realDir ? 4 : 3

    if MType == AHO
        NObs = 6*BT.model.contour.t_steps
    else
        NObs = 4
    end

    if realDir
        LOs = zeros(T,NObs,length(Ys),length(Xs),NTr)
        L2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),length(Xs),NTr)
    else
        LOs = zeros(T,NObs,length(Ys),NTr)
        L2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),NTr)
    end

    Threads.@threads for tr in 1:NTr
    #for tr in 1:NTr
        
        N = size(sol[tr])[2]
        trVs = zeros(T,NObs,N)
        trV2s = (L2 == Nothing) ? nothing : zeros(T,NObs,N)
        
        @inbounds @simd for i in 1:N
            tri = @view sol[tr][:,i]
            trVs[:,i] = BT.L_CO(tri)

            if L2 != Nothing
                trV2s[:,i] = BT.L2_CO(tri)
            end
        end
        

        ### TODO: This needs to be abstracted away to the specific model
        if length(sol[tr][:,1]) > 2
            if realDir
                imX = [maximum(abs.(@view sol[tr][1:div(end,2):end,:]),dims=1);maximum(abs.(@view sol[tr][div(end,2)+1:end,:]),dims=1)]
            else
                imX = maximum(abs.(@view sol[tr][div(end,2)+1:end,:]),dims=1)
            end
        else
            #imX = maximum(abs.(sol[tr]),dims=1)
            if realDir
                imX = abs.(sol[tr])
            else
                imX = abs.(@view sol[tr][2,:])
            end
        end

        if realDir
            trLOs = zeros(T,NObs,length(Ys),length(Xs))
            trL2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys),length(Xs))
        else
            trLOs = zeros(T,NObs,length(Ys))
            trL2Os = (L2 == Nothing) ? nothing : zeros(Float64,NObs,length(Ys))
        end
        for i in 1:length(Ys)
            if realDir
                for j in 1:length(Xs)
                    Hinx = @. (imX[2,:] .<= Ys[i]) .& (imX[1,:] .<= Xs[j])
                    H = @view trVs[:,Hinx]
                    trLOs[:,i,j] = sum(H,dims=2)/N
                    if L2 != Nothing
                        H2 = @view trV2s[:,Hinx]
                        trL2Os[:,i,j] = sum(H2,dims=2)/N
                    end
                end
            else 
                H = @view trVs[:,vec(imX .<= Ys[i])]
                trLOs[:,i] = sum(H,dims=2)/N
                if L2 != Nothing
                    H2 = @view trV2s[:,imX .<= Ys[i]]
                    trL2Os[:,i] = sum(H2,dims=2)/N
                end
            end
        end
        LOs[map((i) -> :,1:(d-1))...,tr] .= trLOs
        if L2 != Nothing
            L2Os[map((i) -> :,1:(d-1))...,tr] .= trL2Os
        end
    end
    if L2 == Nothing
        if witherror
            return dropdims(mean(LOs,dims=d),dims=d) .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr), 
                    nothing
        else
            return dropdims(mean(LOs,dims=d),dims=d),# .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr), 
                    nothing
        end
    else
        return dropdims(mean(LOs,dims=d),dims=d) .± dropdims(std(LOs,dims=d),dims=d) ./ sqrt(NTr),
               dropdims(mean(L2Os,dims=d),dims=d) .± dropdims(std(L2Os,dims=d),dims=d) ./ sqrt(NTr)
    end
end

function calcBoundaryTermsCorrections(sol,BT::BoundaryTerms,lims::Tuple{T, T},lims2::Tuple{T, T}) where {T <: Real}
    isnothing(BT.L2_CO) && return 0.
    L_CO, L2_CO = calcBoundaryTerms(sol,BT)

    # Interpolate the boundary value
    Yinx = findall((x) -> lims[1] <= x <= lims[2], BT.Ys)
    Yinx2 = findall((x) -> lims2[1] <= x <= lims2[2], BT.Ys)
    B1 = weightedmean(L_CO[Yinx])
    B2 = weightedmean(L2_CO[Yinx2])
    return B1, B2
end


