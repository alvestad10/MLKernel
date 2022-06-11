


function get_ab(model::U1,kernel::ConstantKernel)

    @unpack K, dK, K_dK, sqrtK = kernel
    @unpack β, s = model

    function a_func!(du,u,p,t)
        x,y = u
        _A = similar(du)
    
        _A[1] = -β * cos(x) * sinh(y) - s*x
        _A[2] = β * sin(x) * cosh(y) - s*y
        
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

function get_ab(model::U1,kernel::FieldDependentKernel)

    @unpack β, s = model
    @unpack K, dK, K_dK, sqrtK = kernel

    function a_func!(du,u,p,t)
        x,y = u
        _A = similar(du)
    
        _A[1] = -β * cos(x) * sinh(y) - s*x
        _A[2] = β * sin(x) * cosh(y) - s*y
        
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


function calc_obs(KP::KernelProblem{U1},sol)
    t_steps = 1

    T = eltype( getKernelParams(KP.kernel) )
    exp1Re = zeros(T,length(sol),t_steps)
    exp1Im = zeros(T,length(sol),t_steps)
    exp2Re = zeros(T,length(sol),t_steps)
    exp2Im = zeros(T,length(sol),t_steps)

    for i in 1:length(sol)
        _u = hcat(sol[i].u...)

        exp1 = exp.(im*(_u[1:1,:] .+ im*_u[2:2,:]))
        exp2 = exp.(im*2*(_u[1:1,:] .+ im*_u[2:2,:]))

        exp1Re[i,:] .= mean(real(exp1),dims=2)[:,1]
        exp1Im[i,:] .= mean(imag(exp1),dims=2)[:,1]

        exp2Re[i,:] .= mean(real(exp2),dims=2)[:,1]
        exp2Im[i,:] .= mean(imag(exp2),dims=2)[:,1]
    end

    #instabilities = sum([tr.retcode != :Success ? 1 : 0 for tr in sol])^2

    return exp1Re, exp1Im, exp2Re, exp2Im
end

function calc_meanObs(KP::KernelProblem{U1};sol=sol)
    NTr = length(sol)
    obs = calc_obs(KP,sol)
    return calc_meanObs(KP,obs,NTr)
end

function calc_meanObs(::KernelProblem{U1},obs,NTr)
    exp1Re, exp1Im, exp2Re, exp2Im = obs
    d = 1
    return mean(exp1Re,dims=d)[1,:], (std(exp1Re,dims=d)/sqrt(NTr))[1,:], 
           mean(exp1Im,dims=d)[1,:], (std(exp1Im,dims=d)/sqrt(NTr))[1,:],
           mean(exp2Re,dims=d)[1,:], (std(exp2Re,dims=d)/sqrt(NTr))[1,:], 
           mean(exp2Im,dims=d)[1,:], (std(exp2Im,dims=d)/sqrt(NTr))[1,:]
end

function calcTrueLoss(sol,KP::KernelProblem{U1})
        
    obs = calc_obs(KP,sol)
    exp1Re, err_exp1Re, exp1Im, err_exp1Im, exp2Re, err_exp2Re, exp2Im, err_exp2Im = calc_meanObs(KP,obs,length(sol)) 
    
    return sum(abs2,[real(KP.y["exp1Re"]) .- exp1Re; 
                     imag(KP.y["exp1Im"]) .- exp1Im]); #err_avgRe; err_avgIm;
                     #real(KP.y["exp2Re"]) .- exp2Re; #err_avg2Re;
                     #imag(KP.y["exp2Im"]) .- exp2Im])#; err_avg2Im])
end

function calcSymLoss(sol,KP::KernelProblem{U1})
    obs = calc_obs(KP,sol)
    exp1Re, err_exp1Re, exp1Im, err_exp1Im, exp2Re, err_exp2Re, exp2Im, err_exp2Im = calc_meanObs(KP,obs,length(sol)) 
    return sum(abs2,[real(KP.y["exp1Re"]) .- exp1Re; 
                     imag(KP.y["exp1Im"]) .- exp1Im]); #err_avgRe; err_avgIm;
                     #real(KP.y["exp2Re"]) .- exp2Re; #err_avg2Re;
                     #imag(KP.y["exp2Im"]) .- exp2Im])#; err_avg2Im])
end


function calcDriftOpt(sol,KP::KernelProblem{U1,T};p=getKernelParams(KP.kernel)) where {T}

    @unpack β,s = KP.model
    @unpack K, K_dK = KP.kernel

    g(u) = begin
        x,y = u
    
        _ARe = -β * cos(x) * sinh(y) - s*x
        _AIm = β * sin(x) * cosh(y) - s*y
        
        (KRe,KIm),dK = K_dK(u,p)
        if isnothing(dK)
            Dre = KRe*_ARe - KIm*_AIm 
            Dim = KRe*_AIm + KIm*_ARe
        else
            Dre = KRe*_ARe - KIm*_AIm + dK[1]
            Dim = KRe*_AIm + KIm*_ARe + dK[2]
        end

        return ( max(Dim * u[2],0.) / abs(u[2]) ) +
               ( max(Dre * u[1],0.) / abs(u[1]) )
    end

    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]^2 

end

function calcBTLoss(sol,KP::KernelProblem{U1},Ys;p=getKernelParams(KP.kernel))
    BT = getBoundaryTerms(KP;Ys=Ys)
    B1, B2 = calcBoundaryTerms(sol,BT)
    return sum(abs,Measurements.value.(B1[1:2,:]))
end


function Loss(L,KP::KernelProblem{U1},NTr;Ys=[])

    @unpack β,s = KP.model
    
    function getLV(sol)

        #=obs = calc_obs(KP,sol)
        avgRe, _, avgIm, _, avg2Re, _, avg2Im, _, avg3Re, _, avg3Im, _ = calc_meanObs(KP,obs,NTr)
        
        imDrift = (L ∈ [:imDriftSym,:imDrift]) ? calcImDrift(sol,KP) : nothing

        if L ∈ [:BT,:BTSym]
            B1, B2 = [isnothing(B) ? B : Measurements.value.(B) for B in calcBoundaryTerms(sol,BT)];
        else
            B1, B2 = nothing, nothing
        end=#

        return nothing #LossVals_LM_AHO(avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im, imDrift, B1, B2)
    end

    function getg(LVals::LossVals_LM_AHO)

        @unpack avgRe, avgIm, avg2Re, avg2Im, avg3Re, avg3Im, imDrift, B1, B2 = LVals
        if L == :Sym
            #=g_Sym(u,p,t) =  begin
                    return (avgRe[1] * u[1] + avgIm[1] * u[2]) +
                        (avg3Re[1] * ((u[1]^2 - u[2]^2)*u[1] - (2 * u[1] * u[2])*u[2]) + # Real x^3
                        avg3Im[1] * ((u[1]^2 - u[2]^2)*u[2] + (2 * u[1] * u[2])*u[1]))   # imag x^3
                
            end=#
            return nothing#g_Sym

        #=elseif L == :imDrift

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
            return g_True=#
        elseif L ∈ [:Sep,:driftOpt,:driftRWOpt]

            # We are not using the LSS for this
            return nothing
        end

    end

    function LTrain(sol,_KP)
        if L ∈ [:True, :imDriftSym, :imDrift, :Sym, :Sep]
            return calcDriftOpt(sol,_KP)#calcSymLoss(sol,_KP)

        #elseif L == :BTSym
        #    return calcSymLoss(sol,_KP) #+
                    #calculateBVLoss(_KP;Ys=Ys,sol=sol)
        elseif L == :BT
            return nothing#calculateBVLoss(_KP;Ys=Ys,sol=sol)
        elseif L == :driftOpt
            return calcDriftOpt(sol,_KP)
        end
               
    end

    function LTrue(sol,_KP)
        return calcTrueLoss(sol,_KP)
    end

    return Loss_LM_AHO(L,Ys,getLV,getg,LTrain,LTrue)
end