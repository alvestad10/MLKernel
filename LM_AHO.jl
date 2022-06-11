using MLKernel
using Plots
using Measurements
using LinearAlgebra, Statistics
using LaTeXStrings

Ys = collect(0.:0.1:5.0)
Xs = collect(0.:0.1:5.0)

M = LM_AHO([0.,4.0],12.0);
#M = LM_AHO([0.0,4.0],12.0);
KP = ConstantKernelProblem(M);
setKernel!(KP.kernel,
    [-π/2,1.],
    #[real(sqrt(1/(M.σ[1]+M.σ[2]*im))),imag(sqrt(1/(M.σ[1]+M.σ[2]*im)))] #
    #[real(sqrt(exp(-im*20*pi/24))),imag(sqrt(exp(-im*20*pi/24)))]
)

KP = FieldDependentKernelProblem(M;θ=angle(-1 + 4*im));
KP = NNDependentKernelProblem(M;N=6);

@time l,sol = trueLoss(KP,tspan=300,NTr=100);
l

#calculateBVLoss(KP;Ys=[2.,3.,4.,5.],tspan=10,NTr=10)#;sol=sol)


BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
@time B1,B2 = calcBoundaryTerms(sol,BT);


plotB(B1,BT)#;plotType=:surfaceCutOut)
plotB(B1,BT;plotType=:surfaceCutOut)
plotB(B1,BT;plotType=:surfaceAndLines)

plotB(B1,B2,BT)#;plotType=:surfaceCutOut)

function g(p)
    #setKernel!(KP.kernel,p)
    BT = getBoundaryTerms(KP;Ys=Ys,p=p)#,Xs=Xs);
    @time B1,B2 = calcBoundaryTerms(sol,BT;T=eltype(p));
    return sum(abs,Measurements.value.(B1[:,10:end]))
end

g([0.,0.0])
MLKernel.ForwardDiff.gradient(g,[0.,0.])
Measurements.value.(B1[:,20:end])


plot(B1[1,:,div(end,2)])
plot!(B1[1,end,:])
plot!(B1[1,:,div(end,2)])


begin
fig = plot()
for i in 1:10:length(collect(0:0.1:5))
    plot!(fig,B1[1,i,:])
end
fig
end

heatmap(collect(0:0.1:5),collect(0:0.1:5),Measurements.value.(B1[1,:,:]))



lhistory = Dict(:L => Float64[], 
            :LTrue => Float64[], 
            :imDrift => Float64[],
            :LSymDrift => Float64[],
            :BT => Float64[],
            :RT => Float64[],
            :Ks => Vector{Float64}[])

cb(LK::LearnKernel;sol=nothing,addtohistory=false) = begin
    KP = LK.KP
    
    if isnothing(sol)
        sol = run_sim(KP,tspan=LK.tspan_test,NTr=LK.NTr_test)
    end

    LTrain = LK.Loss.LTrain(sol)
    TLoss = LK.Loss.LTrue(sol)
    bt =  MLKernel.calcBTLoss(sol,KP,LK.Loss.Ys)
    rt =  MLKernel.calcRealPosDrift(sol,KP)
    LSymDrift =  MLKernel.calcLSymDrift(sol,KP)
    reD =  MLKernel.calcReDrift(sol,KP)
    imD =  MLKernel.calcImDrift(sol,KP)
    LSym =  MLKernel.calcSymLoss(sol,KP)

    @show LSymDrift

    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, avg3Re, err_avg3Re, avg3Im, err_avg3Im = MLKernel.calc_meanObs(KP;sol=sol)
    @show KP.y["x2"],(avg2Re .± err_avg2Re)[1], (avg2Im .± err_avg2Im)[1]
    @show (avgRe .± err_avgRe)[1], (avgIm .± err_avgIm)[1], (avg3Re .± err_avg3Re)[1], (avg3Im .± err_avg3Im)[1]

    if KP.kernel isa MLKernel.ConstantKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", reDrift: ", reD, ", imDrift: ", imD, ", LSym: ", LSym, ", BT: ", bt, ", RT: ", rt, ", Ks: ", MLKernel.getKernelParams(KP.kernel))
    elseif KP.kernel isa MLKernel.FunctionalKernel || KP.kernel.pK isa MLKernel.LM_AHO_NNKernel || KP.kernel.pK isa MLKernel.LM_AHO_FieldKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", reDrift: ", reD, ", imDrift: ", imD, ", LSym: ", LSym, ", BT: ", bt, ", RT: ", rt)
    end

    if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LTrue],TLoss)
        append!(lhistory[:imDrift],imD)
        append!(lhistory[:LSymDrift],LSymDrift)
        append!(lhistory[:BT],bt)
        append!(lhistory[:RT],rt)
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

KP = ConstantKernelProblem(M);
KP = ConstantKernelProblem(M; kernel=MLKernel.ConstantKernel(MLKernel.LM_AHO_ConstKernel2(M)));
KP = NNDependentKernelProblem(M;N=6);
KP = NNDependentKernel1Problem(M;N=8);
LK = MLKernel.LearnKernel(KP,:driftOpt,10; tspan=50.,NTr=20,tspan_test=50.,NTr_test=20,saveat=0.01,
                          #Ys=[0.6,1.0,1.5,2.,3.,4.],
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
#plot!(fig,lhistory[:LSymDrift])
plot!(fig,lhistory[:BT])
plot!(fig,lhistory[:RT])
end

LK.KP.kernel.pK.re(LK.KP.kernel.pK.ann)([1 + im*1])



using ForwardDiff
using Zygote

sol = run_sim(LK)

size(sol[:][1])

g(_p) = begin
    MLKernel.calcImDrift(sol,LK.KP;p=_p)
end

g(MLKernel.getKernelParams(LK.KP.kernel))
@time ForwardDiff.gradient(g,MLKernel.getKernelParams(LK.KP.kernel))
@time MLKernel.Flux.gradient(g,MLKernel.getKernelParams(LK.KP.kernel))[1]
@time Zygote.gradient(g,MLKernel.getKernelParams(LK.KP.kernel))[1]

begin
sol = run_sim(LK)
g(_p) = begin
    MLKernel.calcImDrift(sol,LK.KP;p=_p)
end
_opt = MLKernel.Flux.ADAM(0.01)
for i in 1:3'
    dKs = ForwardDiff.gradient(g,MLKernel.getKernelParams(LK.KP.kernel))
    MLKernel.Flux.update!(_opt, MLKernel.getKernelParams(LK.KP.kernel), dKs)
    @show g(MLKernel.getKernelParams(LK.KP.kernel)) #MLKernel.getKernelParams(KP.kernel)
end
end




sol

imDrift = zeros(Float64,length(sol))
for i in 1:length(sol)
    _u = hcat(sol[i].u...)

    mm = zeros(Float64,size(_u)[2],2)
    _du = similar(_u[:,1])
    @inbounds @simd for j in 1:size(_u)[2]
        KP.a(_du,_u[:,j],KP.kernel.pK.K,0.)
        mm[j] = _du[2]^2
    end
    imDrift[i] = mean(mm)
end

@show mean(imDrift)

# L: 0.02, BVLoss: 94083, imDrift: 67
# L: 1e-5, BVLoss:    42, imDrift: 1.1
# L: 0.1,  BVLoss:   177, imDrift: 3.03



# 38
# 0.59
# 2.2


@time begin

θs = collect(-pi:0.2:0.0)
@show length(θs)

_Ys=[2.,3.,4.]

fig = plot(yaxis=:log)

for (j,A) in enumerate([1.])

    M = LM_AHO([-0.,4.],12.0);

    L = zeros(length(θs))
    driftOpts = zeros(length(θs))
    RP = zeros(length(θs))
    BT = zeros(length(θs))
    #imdrift = zeros(length(θs))
    #redrift = zeros(length(θs))

    for (i,θ) in enumerate(θs)
        KP = ConstantKernelProblem(M; kernel=MLKernel.ConstantKernel(MLKernel.LM_AHO_ConstKernel2(M;θs=[θ,A])));
        l,sol = trueLoss(KP,tspan=100,NTr=100);
        dopt = MLKernel.calcDriftOpt(sol,KP)
        rp = MLKernel.calcRealPosDrift(sol,KP)
        bt = MLKernel.calcBTLoss(sol,KP,_Ys)
        #imd = MLKernel.calcImDrift(sol,KP)
        #red = MLKernel.calcReDrift(sol,KP)
        L[i] = l
        driftOpts[i] = dopt
        RP[i] = rp
        BT[i] = bt
        #imdrift[i] = imd
        #redrift[i] = red
        print(i)
    end
    println("Done: ", A)

    plot!(fig,θs,L;label=string("L(",A,")"))#,color=j)
    plot!(fig,θs,driftOpts;label="driftOpt")#,color=j)
    plot!(fig,θs,RP;label="RP")#,color=j)
    plot!(fig,θs,BT;label="BT")#,color=j)
    #plot!(fig,θs,imdrift;label="imdrift")
    #plot!(fig,θs,redrift;label="redrift")
end
fig
end

begin
fig = plot(yaxis=:log)
plot!(fig,θs,L;label=string("L"))#,color=j)
plot!(fig,θs,driftOpts;label="driftOpt")#,color=j)
plot!(fig,θs,RT;label="RT")
plot!(fig,θs,BT;label="BT")
#plot!(fig,θs,BT .+ 100*driftOpts;label="BTdriftOpt")
fig
end





f(x) = begin
    cnum = x + 4*im
    return angle(cnum)
end
gg(x) = angle(2.)
x = collect(-5.:0.1:10)
plot(x,f.(x))
plot!(x,gg.(x))