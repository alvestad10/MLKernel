
function get_ab(model::AHO,kernel::ConstantKernel{T}) where {T <: AHOConstantKernelParameters}

    @unpack m, λ, contour = model
    @unpack a, t_steps, κ = contour
    @unpack sqrtK,K = kernel.pK

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]


    function a_func!(du,u,p,t)

        _A = similar(du)

        xR =  @view u[1:div(end,2)]
        xI =  @view u[div(end,2)+1:end]
        #x = xR .+ im .* xI

        pre_fac = (1 / abs(a[1]))
        
        _A[1:t_steps] .= - pre_fac .* 0.5 .* (
            2. .* ( real.(a_m1) .* (xI .- xI[gp1])  .- imag.(a_m1 .- κ).*(xR .- xR[gp1]) ) ./ abs.(a_m1).^2
        .- 2. .* ( real.(a)  .* (xI[gm1] .- xI) .- imag.(a .- κ)  .* (xR[gm1] .- xR) )./ abs.(a).^2
        .- real.(a_m1 .+ a) .* (m .* xI .+ (1/6)*λ .* (-(xI.^3) .+ 3xI.*(xR.^2))) 
        .- imag.(a_m1 .+ a) .* (m .* xR .+ (1/6)*λ .* (xR.^3 .- 3xR.*(xI.^2))) 
        )     
        
        _A[t_steps+1:end] .= pre_fac .* 0.5 .* (
            2. .* (real.(a_m1).*(xR .- xR[gp1]) .+ imag.(a_m1 .- κ).*(xI .- xI[gp1])) ./ abs.(a_m1).^2
        .- 2. .* (real.(a).*(xR[gm1] .- xR)   .+ imag.(a .- κ).*(xI[gm1] .- xI)) ./ abs.(a).^2
        .- real.(a_m1 .+ a) .* (m.*xR .+ (1/6)*λ .* ((xR.^3) .- 3xR.*(xI.^2))) 
        .+ imag.(a_m1 .+ a) .* (m.*xI .+ (1/6)*λ .* (-xI.^3 .+ 3xI.*(xR.^2))) 
        )
        mul!(du,K,_A)

        #A = im*pre_fac*( 
        #        (x .- x[gp1]) ./ a_m1 + (x .- x[gm1]) ./ a
        #        .- (a .+ a_m1)/2 .* (m .* x .+ (λ/6) .* x.^3)
        #)

        #ARe = real(A)
        #AIm = imag(A)

        #mul!(du,K,[ARe ; AIm])


        #=_A[1:t_steps] .= - 0.5 .* (
        #    2. .* ( real.(a_m1) .* (xI .- xI[gp1])  .- imag.(a_m1 .- κ).*(xR .- xR[gp1]) ) ./ abs.(a_m1).^2
        #.- 2. .* ( real.(a)  .* (xI[gm1] .- xI) .- imag.(a .- κ)  .* (xR[gm1] .- xR) )./ abs.(a).^2
        .- real.(a_m1 .+ a) .* ( (1/6)*λ .* (-(xI.^3) .+ 3xI.*(xR.^2))) 
        .- imag.(a_m1 .+ a) .* ( (1/6)*λ .* (xR.^3 .- 3xR.*(xI.^2))) 
        )     
        
        _A[t_steps+1:end] .= 0.5 .* (
        #    2. .* (real.(a_m1).*(xR .- xR[gp1]) .+ imag.(a_m1 .- κ).*(xI .- xI[gp1])) ./ abs.(a_m1).^2
        #.- 2. .* (real.(a).*(xR[gm1] .- xR)   .+ imag.(a .- κ).*(xI[gm1] .- xI)) ./ abs.(a).^2
        .- real.(a_m1 .+ a) .* ( (1/6)*λ .* ((xR.^3) .- 3xR.*(xI.^2))) 
        .+ imag.(a_m1 .+ a) .* ( (1/6)*λ .* (-xI.^3 .+ 3xI.*(xR.^2))) 
        )


        du .= pre_fac * ( -u .+ K*_A)=#

    end
    
    function b_func!(du,u,p,t)
        pre_fac = (1 / abs(a[1]))
        du .= sqrt(2 * pre_fac)*sqrtK
    end 

    return a_func!, b_func!

end

function get_ab(model::AHO,kernel::FieldDependentKernel)

    @unpack m, λ, contour = model
    @unpack a, t_steps, κ = contour
    @unpack sqrtK,K_dK = kernel

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]


    function a_func!(du,u,p,t)

        #_A = similar(du)

        xR =  @view u[1:div(end,2)]
        xI =  @view u[div(end,2)+1:end]

        pre_fac = (1 / abs(a[1]))
        
        _ARe = - pre_fac .* 0.5 .* (
            2. .* ( real.(a_m1) .* (xI .- xI[gp1])  .- imag.(a_m1 .- κ).*(xR .- xR[gp1]) ) ./ abs.(a_m1).^2
        .- 2. .* ( real.(a)  .* (xI[gm1] .- xI) .- imag.(a .- κ)  .* (xR[gm1] .- xR) )./ abs.(a).^2
        .- real.(a_m1 .+ a) .* (m .* xI .+ (1/6)*λ .* (-(xI.^3) .+ 3xI.*(xR.^2))) 
        .- imag.(a_m1 .+ a) .* (m .* xR .+ (1/6)*λ .* (xR.^3 .- 3xR.*(xI.^2))) 
        )     
        
        _AIm = pre_fac .* 0.5 .* (
            2. .* (real.(a_m1).*(xR .- xR[gp1]) .+ imag.(a_m1 .- κ).*(xI .- xI[gp1])) ./ abs.(a_m1).^2
        .- 2. .* (real.(a).*(xR[gm1] .- xR)   .+ imag.(a .- κ).*(xI[gm1] .- xI)) ./ abs.(a).^2
        .- real.(a_m1 .+ a) .* (m.*xR .+ (1/6)*λ .* ((xR.^3) .- 3xR.*(xI.^2))) 
        .+ imag.(a_m1 .+ a) .* (m.*xI .+ (1/6)*λ .* (-xI.^3 .+ 3xI.*(xR.^2))) 
        )

        (KRe,KIm),(dKRe,dKIm) = K_dK(u,p)

        #mul!(du,K,_A)
        du[1:div(end,2),:] .= KRe*_ARe .- KIm*_AIm .+ dKRe
        du[div(end,2)+1:end,:] .= KRe*_AIm .+ KIm*_ARe .+ dKIm
    end
    
    function b_func!(du,u,p,t)
        pre_fac = (1 / abs(a[1]))
        H = sqrtK(u,p)
        du .= sqrt(2 * pre_fac)*H
    end 

    return a_func!, b_func!

end


function calc_obs(KP::KernelProblem{AHO},sol)
    t_steps = KP.model.contour.t_steps

    T = eltype( getKernelParams(KP.kernel) )
    avgRe = zeros(T,length(sol),t_steps)
    avgIm = zeros(T,length(sol),t_steps)
    avg2Re = zeros(T,length(sol),t_steps)
    avg2Im = zeros(T,length(sol),t_steps)
    corr0tRe = zeros(T,length(sol),t_steps)
    corr0tIm = zeros(T,length(sol),t_steps)

    for i in 1:length(sol)
        _u = hcat(sol[i].u...)
        avgRe[i,:] .= mean(_u[1:t_steps,:],dims=2)[:,1]
        avgIm[i,:] .= mean(_u[t_steps+1:end,:],dims=2)[:,1]

        x2Re = _u[1:t_steps,:].^2 .- _u[t_steps+1:end,:].^2
        x2Im = 2 .* _u[1:t_steps,:] .* _u[t_steps+1:end,:]

        avg2Re[i,:] .= mean(x2Re,dims=2)[:,1]
        avg2Im[i,:] .= mean(x2Im,dims=2)[:,1]
        corr0tRe[i,:] .= mean(_u[1:1,:] .* _u[1:t_steps,:] .- _u[t_steps+1:t_steps+1,:].*_u[t_steps+1:end,:],dims=2)[:,1]
        corr0tIm[i,:] .= mean(_u[1:t_steps,:] .* _u[t_steps+1:t_steps+1,:] .+ _u[1:1,:] .* _u[t_steps+1:end,:],dims=2)[:,1]
    end

    #instabilities = sum([tr.retcode != :Success ? 1 : 0 for tr in sol])^2

    return avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm #, corr0tRe, corr0tIm, instabilities
end

function calc_meanObs(::KernelProblem{AHO},obs,NTr)
    avgRe, avgIm, avg2Re, avg2Im, corr0tRe, corr0tIm = obs
    d = 1
    #return mean(avgRe,dims=d)[1,:], (std(avgRe,dims=d)/sqrt(NTr))[1,:], 
    #       mean(avgIm,dims=d)[1,:], (std(avgIm,dims=d)/sqrt(NTr))[1,:],
    #       mean(avg2Re,dims=d)[1,:], (std(avg2Re,dims=d)/sqrt(NTr))[1,:], 
    #       mean(avg2Im,dims=d)[1,:], (std(avg2Im,dims=d)/sqrt(NTr))[1,:],
    #       mean(corr0tRe,dims=d)[1,:], (std(corr0tRe,dims=d)/sqrt(NTr))[1,:], 
    #       mean(corr0tIm,dims=d)[1,:], (std(corr0tRe,dims=d)/sqrt(NTr))[1,:]
    
    return mean(avgRe,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(avgRe')],#(std(avgRe,dims=d)/sqrt(NTr))[1,:], 
           mean(avgIm,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(avgIm')],#(std(avgIm,dims=d)/sqrt(NTr))[1,:],
           mean(avg2Re,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(avg2Re')],#(std(avg2Re,dims=d)/sqrt(NTr))[1,:], 
           mean(avg2Im,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(avg2Im')],#(std(avg2Im,dims=d)/sqrt(NTr))[1,:],
           mean(corr0tRe,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(corr0tRe')],#(std(corr0tRe,dims=d)/sqrt(NTr))[1,:], 
           mean(corr0tIm,dims=d)[1,:], [sqrt(Jackknife.variance(mean,X)) for X in eachrow(corr0tIm')]#(std(corr0tRe,dims=d)/sqrt(NTr))[1,:]
end

function calcTrueLoss(sol,KP::KernelProblem{AHO})
        
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 

    return sum(abs2,[real(KP.y["x"]) .- avgRe; 
                     imag(KP.y["x"]) .- avgIm; #err_avgRe; err_avgIm;
                     real(KP.y["x2"]) .- avg2Re; #err_avg2Re;
                     imag(KP.y["x2"]) .- avg2Im;
                     real(KP.y["corr0t"]) .- corr0tRe; #err_avg2Re;
                     imag(KP.y["corr0t"]) .- corr0tIm])#; err_avg2Im])
end

function calcSymLoss(sol,KP::KernelProblem{AHO})

    βsteps = KP.model.contour.EucledianSteps

    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol)) 
    return sum(abs2,[real(KP.y["x"]) .- avgRe; 
                     imag(KP.y["x"]) .- avgIm; #err_avgRe; err_avgIm;
                     real(KP.y["x2"]) .- avg2Re; #err_avg2Re;
                     imag(KP.y["x2"]) .- avg2Im;
                     real(KP.y["corr0t"][end-βsteps]) .- corr0tRe[end-βsteps]; #err_avg2Re;
                     imag(KP.y["corr0t"][end-βsteps]) .- corr0tIm[end-βsteps]])#; err_avg2Im])
end


function calcDriftOpt(sol,KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel}

    @unpack m, λ, contour = KP.model
    @unpack a, t_steps, κ = contour
    @unpack K, sqrtK = KP.kernel

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]


    KRe,KIm = K([],p)
    #_sqrtK = sqrtK([],p)

    #Δt = 1e-2

    g(u) = begin
        xR =  @view u[1:div(end,2)]
        xI =  @view u[div(end,2)+1:end]

        x = xR + im*xI

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
        
        #A = im * pre_fac * ( 
        #        (x .- x[gp1]) ./ a_m1 + (x .- x[gm1]) ./ a
        #        .- (a .+ a_m1)/2 .* (m .* x .+ (λ/6) .* x.^3)
        #)

        #ARe = real(A)
        #AIm = imag(A)

        
        

        #ξ = randn(t_steps)
        #dWRe = sqrt(2 * Δt * pre_fac)*_sqrtK[1:div(end,2),:] * ξ
        #dWIm = sqrt(2 * Δt * pre_fac)*_sqrtK[div(end,2)+1:end,:] * ξ

        Dre = (KRe*ARe .- KIm*AIm)#*Δt .+ dWRe 
        Dim = (KRe*AIm .+ KIm*ARe)#*Δt .+ dWIm

        #return sum(  (max.((Dim .+ dWIm) .* u[div(end,2)+1:end],0.) ./ abs.(u[div(end,2)+1:end]) )  ) / t_steps +
        #       sum(  (max.((Dre .+ dWRe) .* u[1:div(end,2)],0.) ./ abs.(u[1:div(end,2)]) )  ) / (t_steps)

        #return sum(  (max.(Dim .* u[div(end,2)+1:end],0.) ./ abs.(u[div(end,2)+1:end]) )  ) / t_steps +
        #       sum(  (max.(Dre .* u[1:div(end,2)],0.) ./ abs.(u[1:div(end,2)]) )  ) / (t_steps)

        #return acos((transpose(Dim)*(-xI)) / (norm(Dim)*norm((-xI)))) + acos((transpose(Dre)*(-xR)) / (norm(Dre)*norm((-xR))))
        
        #return acos((transpose(Dim)*(-xI)) / (norm(Dim)*norm((-xI))))^2 + acos((transpose(Dre)*(-xR)) / (norm(Dre)*norm((-xR))))^2
        #return norm([Dre;Dim])*acos((transpose([Dre;Dim])*(-u)) / (norm([Dre;Dim])*norm((-u))))^4

        D = Dre + im*Dim 
        #return mean(norm.(D) .* acos.( (Dre .* (-xR) .+ Dim .* (-xI))  ./ (norm.(D) .* norm.(x))).^2)
        #return mean(abs2, (real(conj.(D) .* (-x)) - norm.(D) .* norm.(x)) ./ (norm.(x)) )
        return mean(abs, (real(conj.(D) .* (-x)) - norm.(D) .* norm.(x)) )
        #return mean(abs, (real(adjoint(D) * (-x)) - norm(D) * norm(x)))

        #return max(transpose([Dre;Dim])*u,0.) / norm(u)
        #return sum(max.([Dre;Dim].*u,0.) / abs.(u)) / t_steps

        #D = Dre .+ im*Dim
        #dW = dWRe .+ im*dWIm
        #x = xR + im*xI
        #x2 = x.^2
        #Dphi2 = x .* (D .+ dW) .+ sum(KRe .+ im*KIm,dims=2)[:]*Δt
        #Dphi2 = x .* D .+ sum(KRe .+ im*KIm,dims=2)[:]

        #return mean(norm.(D) .* acos.( (Dre .* (-xR) .+ Dim .* (-xI))  ./ (norm.(D) .* norm.(U))).^2) +
        #       mean(norm.(Dphi2) .* acos.( (real(Dphi2) .* real(-x2) .+ imag(Dphi2) .* imag(-x2))  ./ (norm.(Dphi2) .* norm.(x2))).^2)

        #return mean(norm.(D) .* acos.( real(conj.(D) .* (-x)) ./ (norm.(D) .* norm.(U))).^2) +
        #       mean(norm.(Dphi2) .* acos.( real(conj(Dphi2) .* (-x2))  ./ (norm.(Dphi2) .* norm.(x2))).^2)

    
        #return mean(abs, (real(adjoint(D) * (-x)) - norm(D) * norm(x))) + # ./ (norm.(x)) ) +
        #       mean(abs, (real(adjoint(Dphi2) * (-x2)) - norm(Dphi2) * norm(x2)))# ./ (norm.(x2)) )



        #return mean(norm.(Dphi2) .* acos.( (real(Dphi2) .* real(-x2) .+ imag(Dphi2) .* imag(-x2))  ./ (norm.(Dphi2) .* norm.(x2))).^2)
        #return sum(max.([real(Dphi2);imag(Dphi2)] .* [real(x2);imag(x2)],0.) / abs.([real(x2);imag(x2)])) / t_steps

    end

    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]#^2

end

function calcDriftOpt(sol,KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T<:FunctionalKernel}
    @unpack m, λ, contour = KP.model
    @unpack a, t_steps, κ = contour
    @unpack sqrtK,K_dK = KP.kernel

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]

    g(u) = begin
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

        (KRe,KIm),(dKRe,dKIm) = K_dK(u,p)

        Dre = KRe*ARe .- KIm*AIm .+ dKRe
        Dim = KRe*AIm .+ KIm*ARe .+ dKIm

        return sum(  (max.(Dim .* u[div(end,2)+1:end],0.) ./ abs.(u[div(end,2)+1:end]) )  ) / t_steps +
               sum(  (max.(Dre .* u[1:div(end,2)],0.) ./ abs.(u[1:div(end,2)]) )  ) / (t_steps)
    end

    return mean(
            mean(
               g(u) for u in eachrow(tr')
            ) for tr in sol)[1]^2
end



function calcDrift(sol,KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T}

    @unpack m, λ, contour = KP.model
    @unpack a, t_steps, κ = contour
    @unpack K = KP.kernel

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]


    KRe,KIm = K([],p)

    g(u) = begin
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

        DRe = KRe*ARe .- KIm*AIm
        DIm = KRe*AIm .+ KIm*ARe

        return [sum(abs,DRe);sum(abs,DIm)] / t_steps       
    end
    f(tr) = mean(g,eachrow(tr'))
    return mean(f,sol[:]) #+ 

end

function sumimevals(KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T}
    #H = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    #H = KP.kernel.sqrtK([],p)[1:div(end,2),:] + im*KP.kernel.sqrtK([],p)[div(end,2)+1:end,:]
    #evals = eigvals(H)
    #return 1000*abs(sum(imag(evals))) #+ 
    #return abs(sum(transpose(H) .- H))

    #=S = 0.
    for evi in evals
        mindiff = 1000.
        for evj in evals
            d = abs(evi - conj(evj))
            if d < mindiff
                mindiff = d
            end
        end
        S += mindiff
    end=#
    #return 50*S
    #Zygote.ignore() do
    #    @show maximum(real(evals))
    #    @show 1e18*abs(sum(H .- transpose(H)))
    #end
    #return 10*S + 1e18*abs(sum(H .- transpose(H))) + 5/(maximum(real(evals)) - minimum(real(evals)) + 1e-3)
    return sum(abs.(p .- transpose(p)))/length(p)
    #return 0.#10*S + (10/maximum(real(evals)))^2
    
    #return 5/(maximum(real(evals)) - minimum(real(evals)) + 1e-3) + 1e10*abs(sum(H .- transpose(H)))
end

function KSym(KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel}
    @unpack K = KP.kernel
    KRe, KIm = K([],p)
    _K = KRe + im*KIm
    return sum(abs.(_K .- transpose(_K)))/length(_K)
end

function KSym(KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel{AHO_ConstKernel_expiP}}
    return sum(abs.(p .- transpose(p)))/length(p)
end

function KSym(KP::KernelProblem{AHO,T};p=getKernelParams(KP.kernel)) where {T <: ConstantKernel{AHO_ConstKernel_expA}}
    P = p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]
    return sum(abs.(P .- adjoint(P)))/length(p)
end


function calcLSymDrift(tr,KP::KernelProblem{AHO};p=getKernelParams(KP.kernel))

    @unpack m, λ, contour = KP.model
    @unpack a, t_steps, κ = contour
    @unpack K,sqrtK = KP.kernel

    gp1=vcat([t_steps],1:t_steps-1)
    gm1=vcat(2:t_steps,[1])
    a_m1 = a[vcat(end,1:end-1)]

    KRe,KIm = K([],p)
    _sqrtK = sqrtK([],p)

    Δt = 1e-5

    g(u) = begin
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

        DRe = KRe*ARe .- KIm*AIm
        DIm = KRe*AIm .+ KIm*ARe

        
        ξ = randn(t_steps)
        dWRe = sqrt(2 * Δt .* pre_fac)*_sqrtK[1:div(end,2),:] * ξ
        dWIm = sqrt(2 * Δt .* pre_fac)*_sqrtK[div(end,2)+1:end,:] * ξ

        xRe = xR .+ DRe*Δt .+ dWRe
        xIm = xI .+ DIm*Δt .+ dWIm
        x2Re = xRe.^2 .- xIm.^2
        x2Im = 2 .* xRe .* xIm
        return hcat(xRe, xIm, x2Re, x2Im)
        
        # K*D = x
        #return abs((Dre - u[1])/u[1]) + abs((Dim - u[2])/u[2]) + abs(dK[1])^2 + abs(dK[2])^2
        
        # K*D*x/abs(x)
        #return sum( max.(Dim .* u[div(end,2)+1:end] .+ u[div(end,2)+1:end].^2,0.) ./ abs.(u[div(end,2)+1:end]) )
        #return maximum(  max.(Dim .* u[div(end,2)+1:end],0.) ./ abs.(u[div(end,2)+1:end])  )
    end

    res = mean(
               g(u) for u in eachrow(tr')
    )

    return sum(abs2,res .-  hcat(zeros(t_steps),zeros(t_steps),real(KP.y["x2"]),imag(KP.y["x2"])))
    
    #sol2 = reduce(hcat,[ reduce(hcat,map(g,eachrow(tr'))) for tr in sol])

    #gg(tr) = sum(g(u) for u in eachrow(tr')) / size(tr)[2]
    #@show sum(sol2)
    #return mean(sol2,dims=2)#sum(abs2, sum(gg(tr) for tr in sol) / length(sol) )# .- [zeros(t_steps)...,zeros(t_steps)...,0.31*ones(t_steps)...,zeros(t_steps)...]) / t_steps
            #(100*abs(det(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]) - 1))^2
    #return ThreadsX.sum(
    #            mean(
    #               g(u) for u in eachrow(tr')
    #            ) for tr in sol[:])[1]/length(sol)

end


function Loss(L,KP::KernelProblem{AHO},NTr;Ys=[])

    @unpack m, λ, contour = KP.model
    
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
            return calcSymLoss(sol,_KP) +
                    calcImDrift(sol,_KP)
        elseif L == :BTSym
            return calcSymLoss(sol,_KP) +
                    calculateBVLoss(_KP;Ys=Ys,sol=sol)
        elseif L == :BT
            return calculateBVLoss(_KP;Ys=Ys,sol=sol)
        elseif L == :driftOpt
            return calcDriftOpt(sol,_KP) #+
                   #sumimevals(_KP)
        elseif L == :driftRWOpt
            return calcRWOpt(sol,_KP) +
                   sumimevals(_KP)
        end
               
    end

    function LTrue(sol,_KP)
        return calcTrueLoss(sol,_KP)
    end

    return Loss_LM_AHO(L,Ys,getLV,getg,LTrain,LTrue)
end