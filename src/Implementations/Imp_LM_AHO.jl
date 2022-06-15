

function get_ab(model::LM_AHO,kernel::ConstantKernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack σ, λ = model

    function a_func!(du,u,p,t)

        _A = similar(du)
    
        _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        
        #KRe = p[1]^2 - p[2]^2
        #KIm = 2*p[1]*p[2]

        (KRe,KIm),dK = K_dK(u,p)
        if isnothing(dK)
            du[1] = KRe*_A[1] - KIm*_A[2] 
            du[2] = KRe*_A[2] + KIm*_A[1]
        else
            du[1] = KRe*_A[1] - KIm*_A[2] + dK[1]
            du[2] = KRe*_A[2] + KIm*_A[1] + dK[2]
        end
    end
    
    function b_func!(du,u,p,t)
        sqrtKRe, sqrtKIm = sqrtK(u,p)
        du[1,1] = sqrt(2)*sqrtKRe
        du[2,1] = sqrt(2)*sqrtKIm
    end 

    return a_func!, b_func!

end


function get_ab(model::LM_AHO,kernel::FieldDependentKernel)

    @unpack σ, λ = model
    @unpack K, dK, K_dK, sqrtK = kernel

    function a_func!(du,u,p,t)

        _A = similar(du)
        _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        
        (KRe,KIm),(dKRe,dKIm) = K_dK(u,p)

        du[1] = KRe*_A[1] - KIm*_A[2] + dKRe
        du[2] = KRe*_A[2] + KIm*_A[1] + dKIm
    end
    
    function b_func!(du,u,p,t)
        
        sqrtKRe, sqrtKIm = sqrtK(u,p)
        du[1,1] = sqrt(2)*sqrtKRe
        du[2,1] = sqrt(2)*sqrtKIm
    end 

    return a_func!, b_func!

end

function calc_obs(KP::KernelProblem{LM_AHO},sol)
    
    t_steps=1
    
    T = eltype(getKernelParams(KP.kernel))
    avgRe = zeros(T,length(sol),t_steps)
    avgIm = zeros(T,length(sol),t_steps)
    avg2Re = zeros(T,length(sol),t_steps)
    avg2Im = zeros(T,length(sol),t_steps)
    avg3Re = zeros(T,length(sol),t_steps)
    avg3Im = zeros(T,length(sol),t_steps)

    for i in 1:length(sol)
        _u = hcat(sol[i].u...)
        avgRe[i,:] .= mean(_u[1:t_steps,:],dims=2)[:,1]
        avgIm[i,:] .= mean(_u[t_steps+1:end,:],dims=2)[:,1]

        x2Re = _u[1:t_steps,:].^2 .- _u[t_steps+1:end,:].^2
        x2Im = 2 .* _u[1:t_steps,:] .* _u[t_steps+1:end,:]

        avg2Re[i,:] .= mean(x2Re,dims=2)[:,1]
        avg2Im[i,:] .= mean(x2Im,dims=2)[:,1]
        avg3Re[i,:] .= mean(x2Re .* _u[1:t_steps,:] - x2Im .* _u[t_steps+1:end,:],dims=2)[:,1]
        avg3Im[i,:] .= mean(x2Re .* _u[t_steps+1:end,:] + x2Im .* _u[1:t_steps,:],dims=2)[:,1]
    end

    #instabilities = sum([tr.retcode != :Success ? 1 : 0 for tr in sol])^2

    return avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im #, corr0tRe, corr0tIm, instabilities
end

function calc_meanObs(KP::KernelProblem{LM_AHO};sol=sol)
    NTr = length(sol)
    obs = calc_obs(KP,sol)
    return calc_meanObs(KP,obs,NTr)
end

function calc_meanObs(::KernelProblem{LM_AHO},obs,NTr)
    avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im = obs
    d = 1
    return mean(avgRe,dims=d)[1,:], (std(avgRe,dims=d)/sqrt(NTr))[1,:], 
           mean(avgIm,dims=d)[1,:], (std(avgIm,dims=d)/sqrt(NTr))[1,:],
           mean(avg2Re,dims=d)[1,:], (std(avg2Re,dims=d)/sqrt(NTr))[1,:], 
           mean(avg2Im,dims=d)[1,:], (std(avg2Im,dims=d)/sqrt(NTr))[1,:],
           mean(avg3Re,dims=d)[1,:], (std(avg3Re,dims=d)/sqrt(NTr))[1,:], 
           mean(avg3Im,dims=d)[1,:], (std(avg3Im,dims=d)/sqrt(NTr))[1,:]
end

function calcTrueLoss(sol,KP::KernelProblem{LM_AHO})
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, avg3Re, err_avg3Re, avg3Im, err_avg3Im = calc_meanObs(KP,obs,length(sol)) 
    
    return sum(abs2,[KP.y["x"][1] .- avgRe; 
                     KP.y["x"][2] .- avgIm; #err_avgRe; err_avgIm;
                     KP.y["x2"][1] .- avg2Re; #err_avg2Re;
                     KP.y["x2"][2] .- avg2Im;
                     KP.y["x3"][1] .- avg3Re; #err_avg2Re;
                     KP.y["x3"][2] .- avg3Im])#; err_avg2Im])
end

function calcSymLoss(sol,KP::KernelProblem{LM_AHO})
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, avg3Re, err_avg3Re, avg3Im, err_avg3Im = calc_meanObs(KP,obs,length(sol)) 
    return sum(abs2,[KP.y["x"][1] .- avgRe; 
                     KP.y["x"][2] .- avgIm;
                     KP.y["x3"][1] .- avg3Re; #err_avg2Re;
                     KP.y["x3"][2] .- avg3Im])#; err_avg2Im])
end

function calcImDrift(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    @unpack σ, λ = KP.model
    @unpack K, dK, K_dK, sqrtK = KP.kernel
    g(u) = begin
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        (KRe,KIm),dK = K_dK(u,p)
        if isnothing(dK)
            return (KRe*_A2 + KIm*_A1)^2
        else
            return (KRe*_A2 + KIm*_A1 + dK[2])^2
        end

    end
    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end

function calcDriftOpt(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    @unpack σ, λ = KP.model
    @unpack K, dK, K_dK, sqrtK = KP.kernel
    g(u) = begin
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        
        (KRe,KIm),dK = K_dK(u,p)
        Dre = 0
        Dim = 0
        if isnothing(dK)
            Dre = KRe*_A1 - KIm*_A2
            Dim = KRe*_A2 + KIm*_A1
        else
            Dre = KRe*_A1 - KIm*_A2 + dK[1]
            Dim = KRe*_A2 + KIm*_A1 + dK[2]
        end

        # K*D = x
        #return abs((Dre - u[1])/u[1]) + abs((Dim - u[2])/u[2]) + abs(dK[1])^2 + abs(dK[2])^2
        
        # K*D*x/abs(x)
        #return abs(Dre - _A1 + u[2])^2 + abs(Dim - _A2 + u[1])
        return ( max( Dim*u[2] , 0.) / abs(u[2]) ) +# ( (Dim*u[1] - Dre*u[2])/sqrt(u[1]^2 + u[2]^2) ) + 
               ( max( Dre*u[1] , 0.) / abs(u[1]) )

        #return ( abs( Dim*u[2]) / abs(u[2]) ) #+# ( (Dim*u[1] - Dre*u[2])/sqrt(u[1]^2 + u[2]^2) ) + 
               #( abs( Dre*u[1]) / abs(u[1]) )
        #if Dim*u[2] > 0 
        #    return (Dim*u[2]/(u[1]^2 + u[2]^2))^2
        #else
            #return ( max(abs(Dim) - abs(u[2]), 0.) / (u[1]^2 + u[2]^2) )^2
        #    return 0. #(abs(Dim/u[2]) - 1 ) / abs(u[2])#( max(abs(Dim) - abs(u[2]), 0.) / (u[1]^2 + u[2]^2) )^2
        #end
    end
    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]^2 #+ (100*min(real(eigvals([cos(p[1]) sin(p[1]) ; -sin(p[1]) cos(p[1])]))[1],0.))^2
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end


function calcLSymDrift(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))


    dt = 1e-3

    @unpack σ, λ = KP.model
    @unpack K, dK, K_dK, sqrtK = KP.kernel
    g(u) = begin
        
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        
        (KRe,KIm),dK = K_dK(u,p)
        Dre = 0
        Dim = 0
        if isnothing(dK)
            Dre = KRe*_A1 - KIm*_A2
            Dim = KRe*_A2 + KIm*_A1
        else
            Dre = KRe*_A1 - KIm*_A2 + dK[1]
            Dim = KRe*_A2 + KIm*_A1 + dK[2]
        end


        _up1Re = u[1] + Dre*dt
        _up1Im = u[2] + Dim*dt

        return [_up1Re,_up1Im,_up1Re^2 - _up1Im^2,2*_up1Re*_up1Im]

    end
    res = mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)
    return sum(abs2,res .-  [0.,0.,KP.y["x2"][1],0.])
            

end


function calcRealPosDrift(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    @unpack σ, λ = KP.model
    @unpack K, dK, K_dK, sqrtK = KP.kernel
    g(u) = begin
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        
        (KRe,KIm),dK = K_dK(u,p)
        Dre = 0
        Dim = 0
        if isnothing(dK)
            Dre = KRe*_A1 - KIm*_A2
            Dim = KRe*_A2 + KIm*_A1
        else
            Dre = KRe*_A1 - KIm*_A2 + dK[1]
            Dim = KRe*_A2 + KIm*_A1 + dK[2]
        end

        ##dd = (Dre + im*Dim)/(u[1] + im*u[2])

        # K*D = x
        #return abs((Dre - u[1])/u[1]) + abs((Dim - u[2])/u[2]) + abs(dK[1])^2 + abs(dK[2])^2
        
        # K*D*x/abs(x)
        #return abs(Dre - _A1 + u[2])^2 + abs(Dim - _A2 + u[1])
        
        return abs( Dre - u[1])^2 + abs( Dim - u[2])^2 

        #M = u[1]^2 + u[2]^2
        #return abs(min( (Dre*u[1] + Dim*u[2])/M , 0.)) + ( (Dim*u[1] - Dre*u[2])/M )^2
        #if Dim*u[2] > 0 
        #    return (Dim*u[2]/(u[1]^2 + u[2]^2))^2
        #else
            #return ( max(abs(Dim) - abs(u[2]), 0.) / (u[1]^2 + u[2]^2) )^2
        #    return 0. #(abs(Dim/u[2]) - 1 ) / abs(u[2])#( max(abs(Dim) - abs(u[2]), 0.) / (u[1]^2 + u[2]^2) )^2
        #end
    end
    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]^2 #+ (100*min(real(eigvals([cos(p[1]) sin(p[1]) ; -sin(p[1]) cos(p[1])]))[1],0.))^2
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end

function calcReDrift(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    @unpack σ, λ = KP.model
    @unpack K, dK, K_dK, sqrtK = KP.kernel
    g(u) = begin
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        (KRe,KIm),dK = K_dK(u,p)
        if isnothing(dK)
            return (KRe*_A1 - KIm*_A2)^2
        else
            return (KRe*_A1 - KIm*_A2 + dK[1])^2
        end

    end
    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end


function calcDrift(sol,KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    drift_im= calcImDrift(sol,KP;p=p)
    #=gre(u) = begin
        _A1 = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
        _A2 = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))
        (KRe,KIm),dK = K_dK(u,p)
        if isnothing(dK)
            return (KRe*_A1 - KIm*_A2)^2
        else
            return (KRe*_A1 - KIm*_A2 + dK[1])^2
        end
    end
    drift_re =  calcReDrift(sol,KP;p=p)=#
    return drift_im
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end

function calcLFP(KP::KernelProblem{LM_AHO};p=getKernelParams(KP.kernel))

    σ = KP.model.σ
    λ = KP.model.λ
    _σ = σ[1] + im*σ[2]
    Hfp(u) = begin
        return -(_σ^2 / 4)*u^2 + (3*(λ/6)/2)*u^2 - (_σ*(λ/6)/2)u^4 - ((λ/6)^2/4)*u^6 + (_σ/2) 
    end

    n = 30000
    xmin = -15; xmax = 15
    x = collect(range(xmin,xmax;length=n))
    dx = x[2]-x[1]
    dS = BandedMatrix{ComplexF64}(undef,(n,n), (0,0))
    D2x = BandedMatrix{Float64}(undef,(n,n), (1,1))
    
    Hfp = Hfp.(x)
    dS[band(0)] .= Hfp; 
    D2x[band(0)] .= -2. /(dx)^2; 
    D2x[band(1)] .= 1. /(dx)^2 ; 
    D2x[band(-1)] .= 1. /(dx)^2;
    D2x[band(n)] .= 1. /(dx)^2; 
    D2x[band(-n)] .= 1. /(dx)^2;



    KRe, KIm = KP.kernel.K([],p)
    A = - (KRe + im*KIm) .* (D2x + dS)

    vals,_ = KrylovKit.eigsolve(A,1,:SR)
    return sum(abs,val for val in real(vals) if val < 0)

end

function calcBTLoss(sol,KP::KernelProblem{LM_AHO},Ys;p=getKernelParams(KP.kernel))
    BT = getBoundaryTerms(KP;Ys=Ys)
    B1, B2 = calcBoundaryTerms(sol,BT)
    return sum(abs,Measurements.value.(B1))
end

function Loss(L,KP::KernelProblem{LM_AHO},NTr;Ys=[])

    BT = getBoundaryTerms(KP;Ys=Ys)
    @unpack σ, λ = KP.model
    @unpack K_dK = KP.kernel
    
    function getLV(sol)

        obs = calc_obs(KP,sol)
        avgRe, _, avgIm, _, avg2Re, _, avg2Im, _, avg3Re, _, avg3Im, _ = calc_meanObs(KP,obs,NTr)
        
        imDrift = (L ∈ [:imDriftSym,:imDrift]) ? calcImDrift(sol,KP) : nothing

        if L ∈ [:BT,:BTSym]
            B1, B2 = [isnothing(B) ? B : Measurements.value.(B) for B in calcBoundaryTerms(sol,BT)];
        else
            B1, B2 = nothing, nothing
        end

        return LossVals_LM_AHO(avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im, imDrift, B1, B2)
    end

    function getg(LVals::LossVals_LM_AHO)

        @unpack avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im, imDrift, B1, B2 = LVals
        if L == :Sym
            g_Sym(u,p,t) =  begin
                    return (avgRe[1] * u[1] + avgIm[1] * u[2]) +
                        (avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2]) + # Real x^3
                        avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1]))   # imag x^3
                
            end
            return g_Sym

        elseif L == :imDrift

            g_imDrift(u,p,t) =  begin
                _A = similar(u)
    
                _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
                _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))

                (KRe,KIm),dK = K_dK(u,p)

                if isnothing(dK)
                        return imDrift*(KRe*_A[2] + KIm*_A[1])^2
                else
                        return imDrift*(KRe*_A[2] + KIm*_A[1] + dK[2])^2
                end
            end
            return g_imDrift

        elseif L == :imDriftSym

            g_imDriftSym(u,p,t) =  begin
                _A = similar(u)
    
                _A[1] = -(σ[1] * u[1] - σ[2] * u[2] + (1/6)*λ * (u[1]*(u[1]*u[1] - u[2]*u[2]) - 2*u[1]*u[2]*u[2]))
                _A[2] = -(σ[1] * u[2] + σ[2] * u[1] + (1/6)*λ * (u[2]*(u[1]*u[1] - u[2]*u[2]) + 2*u[1]*u[1]*u[2]))

                (KRe,KIm),dK = K_dK(u,p)

                G = 1000

                if isnothing(dK)
                    return G*(avgRe[1] * u[1] + avgIm[1] * u[2]) +
                           #(avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2]) + # Real x^3
                           # avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1])) + # imag x^3
                            imDrift*(KRe*_A[2] + KIm*_A[1])^2
                else
                    return G*(avgRe[1] * u[1] + avgIm[1] * u[2]) +
                             #(avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2])) + # Real x^3
                             # avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1])) + # imag x^3
                             imDrift*(KRe*_A[2] + KIm*_A[1] + dK[2])^2
                end
            end
            return g_imDriftSym

        elseif L == :BTSym
            g_BTSym(u,p,t) =  begin
                
                R = 0.
                if u[2] < Ys[end]
                    LCO = BT.L_CO(u) 
                    #L2CO = BT.L2_CO(u) 
                    for (i,Y) in enumerate(Ys)
                        u[2] > Y && continue
                        R +=  (B1[1,i] * LCO[1] + B1[2,i] * LCO[2] + B1[3,i] * LCO[3] + B1[4,i] * LCO[4])
                        #R +=  (B2[1,i] * L2CO[1] + B2[2,i] * L2CO[2] + B2[3,i] * L2CO[3] + B2[4,i] * L2CO[4])
                    end
                end

                G = 1000.
                return G*(avgRe[1] * u[1] + avgIm[1] * u[2]) +
                       # (avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2]) + # Real x^3
                       # avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1])) + # imag x^3
                        (R / (sum(Ys)))
                
            end
            return g_BTSym
        elseif L == :BT
            g_BT(u,p,t) =  begin
                
                R = 0.
                if u[2] < Ys[end]
                    LCO = BT.L_CO(u) 
                    #L2CO = BT.L2_CO(u) 
                    for (i,Y) in enumerate(Ys)
                        u[2] > Y && continue
                        R +=  Y*(B1[1,i] * LCO[1] + B1[2,i] * LCO[2] + B1[3,i] * LCO[3] + B1[4,i] * LCO[4])
                        #R +=  (B2[1,i] * L2CO[1] + B2[2,i] * L2CO[2] + B2[3,i] * L2CO[3] + B2[4,i] * L2CO[4])
                    end
                end

                return (R / sum(Ys))
                
            end
            return g_BT
        elseif L == :True
            g_True(u,p,t) =  begin
                return  avgRe[1] * u[1] + avgIm[1] * u[2] +
                    (KP.y["x2"][1] - avg2Re[1]) * (KP.y["x2"][1] - (u[1]^2 - u[2]^2)) + 
                    (KP.y["x2"][2] - avg2Im[1]) * (KP.y["x2"][2] - 2 * u[1] * u[2]) +
                    (avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2]) + # Real x^3
                     avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1]))   # imag x^3
            end
            return g_True
        elseif L == :Sep

            # We are not using the LSS for this
            return nothing
        end

    end

    function LTrain(sol)

        if L ∈ [:imDriftSym, :imDrift, :Sym, :Sep]
            return calcSymLoss(sol,KP) +
                    calcImDrift(sol,KP)
        elseif L == :BTSym
            return calcSymLoss(sol,KP) +
                    calculateBVLoss(KP;Ys=Ys,sol=sol)
        elseif L == :BT
            return calculateBVLoss(KP;Ys=Ys,sol=sol)
        elseif L ∈ [:True, :driftOpt]
            return calcDriftOpt(sol,KP)
        elseif L ∈ [:Sym, :Sep]
            return calcSymLoss(sol,KP) + 
                   calcDriftOpt(sol,KP)
        elseif L ∈ [:FP]
            return calcDriftOpt(sol,KP)
        end
               
    end

    function LTrue(sol)
        return calcTrueLoss(sol,KP)
    end

    return Loss_LM_AHO(L,Ys,getLV,getg,LTrain,LTrue)
end