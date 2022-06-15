export LearnKernel, learnKernel

mutable struct LearnKernel{LType<:AbstractLoss}
    KP::KernelProblem
    Loss::LType
    
    # Optimization parameters
    opt::Flux.Optimise.AbstractOptimiser
    epochs::Integer
    runs_pr_epoch::Integer
    
    # Simulation parameters
    tspan::Float64
    saveat::Float64
    NTr::Integer

    tspan_test::Float64
    NTr_test::Integer

    # LSS parameters
    alpha::Float64
end

function LearnKernel(KP::KernelProblem,L::Symbol,epochs;runs_pr_epoch=1,tspan=10.,tspan_test=tspan,saveat=0.02,NTr=10,NTr_test=NTr,alpha=10.,Ys=[1.,2.],opt=ADAM(0.05))
    L = Loss(L,KP,NTr;Ys=Ys)

    return LearnKernel(KP,L,opt,epochs,runs_pr_epoch,tspan,saveat,NTr,tspan_test,NTr_test,alpha)
end

function updatelr!(LK::LearnKernel,lr)
    LK.opt.eta = lr
end


function StartingLKText(LK::LearnKernel)
    @unpack epochs, runs_pr_epoch, alpha, NTr, tspan, saveat, Loss = LK
    
    println("*************************************")
    println("***** Starting learning kernel ******")
    println("  Type of loss function: ", Loss.ID)
    println("  Total epochs is ", epochs, " with ", runs_pr_epoch, " runs pr. epoch")
    println("  alpha=", alpha)
    println("  NTr=", NTr)
    println("  tspan=", tspan)
    println("  saveat=", saveat)
    println("*************************************")
end

##### LearnKernel shorthand functions ####

function calculateBVLoss(LK::LearnKernel;sol=nothing)
    @unpack KP, Loss, tspan, NTr = LK
    return calculateBVLoss(KP; Ys=Loss.Ys, tspan=tspan, NTr=NTr, sol=sol)
end

function run_sim(LK::LearnKernel)
    return run_sim(LK.KP;tspan=LK.tspan,NTr=LK.NTr,saveat=LK.saveat)
end

"""
   CHECK FOR ERRORS/WARNINGS DURING RUN

   can also remove warning trajectories by setting
                remove_warning_tr=true
"""
function check_warnings!(sol) #;remove_warning_tr=true)
    #allok = true
    warnings_inxes = []
    for (i,s) in enumerate(sol) 
        if (s.retcode != :Success)
            #allok = false
            #println("Trajectory ",i," has retcode: ", string(s.retcode))
            push!(warnings_inxes,i) 

        end
    end
        
    deleteat!(sol.u,warnings_inxes)
    return isempty(sol)

    
    #if remove_warning_tr && !allok
        #println("Removing trajectories ", warnings_inxes)
    #else
        #println("All ok")
    #end 


end


#########

function dL(LK::LearnKernel;sol=nothing)

    @unpack KP, Loss, tspan, NTr,saveat,alpha = LK
    if isnothing(sol)
        sol = run_sim(KP;tspan=tspan,NTr=NTr,saveat=saveat)
    end
    
    LVals = Loss.get_LV(sol)

    function get_derivative_of_trajectory(tr)
        
        g = Loss.get_g(LVals) 

        lss_problem = AdjointLSSProblem(tr, AdjointLSS(alpha=alpha), g; autojacvec = ReverseDiffVJP(true))
        shadow_adjoint(lss_problem)
    end

    #res = map(get_derivative_of_trajectory,sol)
    res = ThreadTools.tmap(get_derivative_of_trajectory,6,sol)

    return 2*mean(res),std(2. * res)/sqrt(NTr)
end

function d_driftOpt(LK::LearnKernel;sol=nothing,ID = :driftOpt)
    
    @unpack KP, Loss, tspan, NTr,saveat,alpha = LK
    
    if isnothing(sol)
        sol = run_sim(KP;tspan=tspan,NTr=NTr,saveat=saveat)
    end

    g = if ID == :driftOpt
        gDriftOpt(_p,i) = MLKernel.calcDriftOpt(sol[i:i],KP;p=_p)
    elseif ID == :BTOpt
        @warn "BTOpt not implemented"        
        gBTOpt(_p,i) = 0.
    elseif ID == :LSymOpt
        gLSYmOpt(_p,i) = MLKernel.calcLSymDrift(reduce(hcat,sol[i].u),KP;p=_p) #+ 
    end

    if length(MLKernel.getKernelParams(KP.kernel)) > 100
        return ThreadsX.sum((i) -> Zygote.gradient((p) -> g(p,i),MLKernel.getKernelParams(KP.kernel))[1],1:length(sol)) ./ length(sol) #.+
           #Zygote.gradient((_p) -> KSym(LK.KP;p=_p),MLKernel.getKernelParams(KP.kernel))[1]
    else
        #return ThreadsX.sum((i) -> ForwardDiff.gradient((p) -> g(p,i),MLKernel.getKernelParams(KP.kernel)),1:length(sol)) ./ length(sol)
        return ForwardDiff.gradient((p) -> MLKernel.calcDriftOpt(sol,KP;p=p),MLKernel.getKernelParams(KP.kernel))
    end
    #Zygote.gradient((p) -> 20*abs(det(p[1:div(end,2),:] + im*p[div(end,2)+1:end,:]) - 1)^2,MLKernel.getKernelParams(KP.kernel))
    
    
end

function d_FP(LK::LearnKernel)
    
    @unpack KP = LK
    #return Zygote.gradient((p) -> calcLFP(KP;p=p),MLKernel.getKernelParams(KP.kernel))[1]
    #return ForwardDiff.gradient((p) -> calcLFP(KP;p=p),MLKernel.getKernelParams(KP.kernel))
    d = FiniteDifferences.grad(central_fdm(3, 1), (p) -> calcLFP(KP;p=p), MLKernel.getKernelParams(KP.kernel))
    #@show d
    return d[1]
end


function dBTOpt(LK::LearnKernel; sol=nothing)

    @unpack KP, Loss, tspan, NTr,saveat,alpha = LK
    @unpack Ys = Loss

    if isnothing(sol)
        sol = run_sim(KP;tspan=tspan,NTr=NTr,saveat=saveat)
    end

    function g(_p)
        #setKernel!(KP.kernel,p)
        BT = getBoundaryTerms(KP;Ys=Ys,p=_p)#,Xs=Xs);
        B1,B2 = calcBoundaryTerms(sol,BT;T=eltype(_p),witherror=false);
        return sum(abs,Measurements.value.(B1)) + MLKernel.calcDriftOpt(sol,KP;p=_p)
    end

    return ForwardDiff.gradient((p) -> g(p),MLKernel.getKernelParams(KP.kernel))
end


function learnKernel(LK::LearnKernel; cb=(LK::LearnKernel; sol=nothing, addtohistory=false) -> ())

    @unpack KP, opt, epochs, tspan, NTr, saveat, runs_pr_epoch = LK
    
    # Use to see if more configurations is needed for the testset
    test_eq_train = (tspan == LK.tspan_test && NTr == LK.NTr_test)

    ### Text to the user
    StartingLKText(LK)

    if KP.kernel isa ConstantKernel
        LK.KP = updateProblem(KP)
    end

    ###### Getting initial configurations
    trun = @elapsed sol = run_sim(LK.KP;tspan=tspan,NTr=NTr,saveat=saveat)
    if check_warnings!(sol)
        @warn "All trajectories diverged"
        return 0
    end
    cb(LK;sol = (test_eq_train ? sol : nothing),addtohistory=true)
    
    # initialize the derivative observable
    dKs = similar(getKernelParams(LK.KP.kernel))
    
    
    for i in 1:epochs
        
        unstable = false

        println("EPOCH ", i, "/", epochs)#+start_inx)
        
        for j in 1:runs_pr_epoch
            
            gotDerivative = false
            
            while !gotDerivative
                

                tdL = @elapsed if LK.Loss.ID ∈ [:True]
                    dKs = dL(LK; sol=sol)[1]
                elseif LK.Loss.ID ∈ [:driftOpt,:BTOpt,:LSymOpt]
                    dKs = d_driftOpt(LK;sol=sol,ID=LK.Loss.ID)
                elseif LK.Loss.ID ∈ [:FP]
                    dKs = d_FP(LK)
                    #@show dKs
                end

                #=tdL = @elapsed begin
                    if LK.Loss.ID ∈ [:Sep,:driftOpt,:driftRWOpt]
                        dKs = 0
                    else
                        dKs = dL(LK; sol=sol)[1]
                    end
                end
                
                if LK.Loss.ID ∈ [:Sep,:driftOpt,:driftRWOpt]
                    tdL2 = @elapsed begin
                        dImDrift = d_imDrift(LK;sol=sol) #+ dBTOpt(LK;sol=sol)
                        #dImDrift = dBTOpt(LK;sol=sol)
                        #id = calcImDrift(sol,KP)
                    end
                    #dKs = ( dKs .+ (dImDrift/100) )/ 2
                    dKs = ( dKs .+ dImDrift )
                    println("Time ", j, ": ", trun, ", ", tdL, ", ", tdL2)
                else
                    println("Time ", j, ": ", trun, ", ", tdL)
                end=#
                println("Time ", j, ": ", trun, ", ", tdL)


                if !any(isnan.(dKs))
                    #@show any(isnan.(dKs))
                    gotDerivative = true
                else
                    tspan += 1
                    println("Detecting NaN in dKs; increasing tspan=", tspan, " and trying again")
                end
            end


            # Updating the kernel parameters
            Flux.update!(opt, getKernelParams(LK.KP.kernel), dKs)

            if KP.kernel isa ConstantKernel
                LK.KP = updateProblem(KP)
            end
            

            trun = @elapsed sol = run_sim(LK.KP;tspan=tspan,NTr=NTr,saveat=saveat)
            if check_warnings!(sol)
                @warn "All trajectories diverged"
                unstable = true
                break
            end

            if runs_pr_epoch == 1
                cb(LK;sol = (test_eq_train ? sol : nothing), addtohistory = (runs_pr_epoch == 1))
            end
        end

        if unstable
            break
        end

        if runs_pr_epoch > 1
            cb(LK; addtohistory=true)
        end
    end


end