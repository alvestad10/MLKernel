using MLKernel
using Plots
using Measurements
using LinearAlgebra, Statistics
using LaTeXStrings

M = U1(0.1,0.0);

Ys = collect(0.:0.1:5.0)
Xs = collect(0.:0.1:5.0)

KP = ConstantKernelProblem(M);
KP = NNDependentKernel1Problem(M;N=8);
@time l,sol = trueLoss(KP,tspan=50,NTr=50);
l

function get_new_lhistory()
return Dict(:L => Float64[], 
            :LTrue => Float64[], 
            :imDrift => Float64[],
            :BT => Float64[],
            :RT => Float64[],
            :Ks => Vector{Float64}[])
end

cb(LK::LearnKernel;sol=nothing,addtohistory=false) = begin
    KP = LK.KP
    
    if isnothing(sol)
        sol = run_sim(KP,tspan=LK.tspan_test,NTr=LK.NTr_test)
    end

    LTrain = LK.Loss.LTrain(sol,KP)
    TLoss = LK.Loss.LTrue(sol,KP)
    bt = MLKernel.calcBTLoss(sol,KP,Ys)

    exp1Re, err_exp1Re, exp1Im, err_exp1Im, exp2Re, err_exp2Re, exp2Im, err_exp2Im = MLKernel.calc_meanObs(KP;sol=sol)
    @show KP.y["exp1Re"],(exp1Re .± err_exp1Re)[1], KP.y["exp1Im"], (exp1Im .± err_exp1Im)[1]
    @show KP.y["exp2Re"],(exp2Re .± err_exp2Re)[1], KP.y["exp2Im"], (exp2Im .± err_exp2Im)[1]
    #@show (avgRe .± err_avgRe)[1], (avgIm .± err_avgIm)[1], (avg3Re .± err_avg3Re)[1], (avg3Im .± err_avg3Im)[1]

    if KP.kernel isa MLKernel.ConstantKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", BT: ", bt, ", K: ", MLKernel.getKernelParams(KP.kernel))
    elseif KP.kernel isa MLKernel.FunctionalKernel || KP.kernel.pK isa MLKernel.LM_AHO_NNKernel || KP.kernel.pK isa MLKernel.LM_AHO_FieldKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", BT: ", bt)
    end

    if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LTrue],TLoss)
        append!(lhistory[:BT],bt)
        if KP.kernel isa MLKernel.AHO_ConstKernel
            append!(lhistory[:Ks],[KP.kernel.pK.K])
        end
    end

    if LK.Loss.ID ∈ [:BT, :BTSym, :driftOpt]
        BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
        B1,B2 = calcBoundaryTerms(sol,BT;witherror=true);
        display(plotB(B1,BT))
    end

end

lhistory = get_new_lhistory()

KP = ConstantKernelProblem(M);
KP = NNDependentKernelProblem(M;N=8);
KP = NNDependentKernel1Problem(M;N=30);
LK = MLKernel.LearnKernel(KP,:driftOpt,10; tspan=30.,NTr=20,tspan_test=30.,NTr_test=20,saveat=0.01,
                          Ys=[1.0,1.5,2.,3.,4.],
                          imDriftSep=true,
                          opt=MLKernel.ADAM(0.05),
                          runs_pr_epoch=1);#, Ys=[1.,1.5,2.,4.,5.])
learnKernel(LK; cb=cb)
cb(LK)

begin
    fig = plot(xlabel="Iteration",yaxis=:log,legend=false)#,ylim=[1,1e6])
    #plot!(fig,lhistory[:L],color=3)
    plot!(fig,lhistory[:LTrue])
    plot!(fig,lhistory[:L])
    plot!(fig,lhistory[:BT])
end




@time l,sol = trueLoss(LK.KP,tspan=30,NTr=20);
l
BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
@time B1,B2 = calcBoundaryTerms(sol,BT);
plotB(B1,BT)


@time begin

    θs = collect(-π:0.1:π)
    @show length(θs)
    
    _Ys=[1.0,1.5,2.,3.,4.]
    
    fig = plot(yaxis=:log)
    
    for (j,A) in enumerate([1.])
    
        M = U1(0.1,0.);
    
        L = zeros(length(θs))
        driftOpts = zeros(length(θs))
        RP = zeros(length(θs))
        BT = zeros(length(θs))
        #imdrift = zeros(length(θs))
        #redrift = zeros(length(θs))
    
        for (i,θ) in enumerate(θs)
            KP = ConstantKernelProblem(M; kernel=MLKernel.ConstantKernel(MLKernel.U1_ConstKernel(M;θs=[θ,A])));
            l,sol = trueLoss(KP,tspan=30,NTr=20);
            dopt = MLKernel.calcDriftOpt(sol,KP)
            L[i] = l
            driftOpts[i] = dopt
            print(i)
        end
        println("Done: ", A)
    
        plot!(fig,θs,L;label=string("L(",A,")"))#,color=j)
        plot!(fig,θs,driftOpts;label="driftOpt")#,color=j)
        #plot!(fig,θs,imdrift;label="imdrift")
        #plot!(fig,θs,redrift;label="redrift")
    end
    fig
    end