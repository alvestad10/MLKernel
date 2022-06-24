using MLKernel
using Plots
using Measurements
using LinearAlgebra, Statistics
using JLD2
using SparseArrays
using LaTeXStrings

Ys = collect(0:0.05:3)
Xs = collect(0:0.05:5)

M=AHO(1.,24.,1.5,1.0,8)#;Δβ=0.5)#,κ=1e-3*im)
scatter(M.contour.x0)

KP = ConstantKernelProblem(M)
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantFreeKernel(M;m=0.5,g=1.0));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel2(M));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel3(M));

# Set free kernel
spy(MLKernel.getKernelParams(KP.kernel), markersize=5)
spy(MLKernel.getKernelParams(KP.kernel) - [Diagonal(ones(M.contour.t_steps));zeros(M.contour.t_steps,M.contour.t_steps)],markersize=3)

spy(LK.KP.kernel.pK.K .- Diagonal(ones(2*M.contour.t_steps)),markersize=5)
spy(LK.KP.kernel.pK.sqrtK .- [Diagonal(ones(LK.KP.model.contour.t_steps));zeros(LK.KP.model.contour.t_steps,LK.KP.model.contour.t_steps)], markersize=5)
spy(LK.KP.kernel.pK.p,markersize=6)

@time l , sol = trueLoss(LK.KP,tspan=30.,NTr=30,saveat=0.01);
l

MLKernel.plotSKContour(LK.KP,sol)
MLKernel.plotFWSKContour(KP,KP,sol)

MLKernel.calcDriftOpt(sol,KP)

function get_new_lhistory()
    return Dict(:L => Float64[], 
    :LTrue => Float64[], 
    :LSymDrift => Float64[], 
    :detK => ComplexF64[],
    :KSym => Float64[],
    :LSym => Float64[])
end

cb(LK::LearnKernel;sol=nothing,addtohistory=false) = begin
    KP = LK.KP
    
    if isnothing(sol)
        sol = run_sim(KP,tspan=LK.tspan_test,NTr=LK.NTr_test)
        if MLKernel.check_warnings!(sol)
            @warn "All trajectories diverged"
        end
    end

    LTrain = LK.Loss.LTrain(sol,KP)
    TLoss = LK.Loss.LTrue(sol,KP)
    reD,imD =  MLKernel.calcDrift(sol,KP)
    #LSymDrift =  mean(MLKernel.calcLSymDrift(reduce(hcat,sol[i].u),KP) for i in 1:LK.NTr)
    #imD =  MLKernel.calcImDrift(sol,KP)
    LSym =  MLKernel.calcSymLoss(sol,KP)

    #magH = det(KP.kernel.pK.sqrtK[1:div(end,2),:] + im*KP.kernel.pK.sqrtK[div(end,2)+1:end,:])
    magH = det(KP.kernel.pK.K[1:div(end,2),1:div(end,2)] + im*KP.kernel.pK.K[div(end,2)+1:end,1:div(end,2)])

    KSym = MLKernel.KSym(KP)
    #avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, avg3Re, err_avg3Re, avg3Im, err_avg3Im = MLKernel.calc_meanObs(KP;sol=sol)
    #@show KP.y["x2"],(avg2Re .± err_avg2Re)[1], (avg2Im .± err_avg2Im)[1]
    #@show (avgRe .± err_avgRe)[1], (avgIm .± err_avgIm)[1], (avg3Re .± err_avg3Re)[1], (avg3Im .± err_avg3Im)[1]

    #if KP.kernel isa MLKernel.ConstantKernel
        #println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", reDrift: ", reD, ", imDrift: ", imD, ", LSym: ", LSym, ", Ks: ", MLKernel.getKernelParams(KP.kernel))
    #elseif KP.kernel.pK isa MLKernel.LM_AHO_NNKernel
    println("LTrain: ", round(LTrain,digits=5), ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5),
            ", reD:", round(reD,digits=5),", imD:", round(imD,digits=5), 
            #", LSymDrift: ",round(LSymDrift,digits=5),
            ", magK: ", round(magH,digits=5),
            ", KSym: ", round(KSym,digits=5))#, ", reDrift: ", reD, ", imDrift: ", imD, ", LSym: ", LSym)
    #end

    display(MLKernel.plotSKContour(KP,sol))
    #display(MLKernel.plotFWSKContour(KP,KP,sol))

    if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LTrue],TLoss)
        append!(lhistory[:LSym],LSym)
        #append!(lhistory[:LSymDrift],LSymDrift)
        append!(lhistory[:detK],magH)
        append!(lhistory[:KSym],KSym)        
    end

    #=if LK.Loss.ID ∈ [:BT, :BTSym, :driftOpt]
        BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
        B1,B2 = calcBoundaryTerms(sol,BT);
        display(plotB(B1,BT))
    end=#

end

lhistory = get_new_lhistory()

KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:K));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:H));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:expiP));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:expA));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:HexpA));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:invM_expiP));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:expiHerm));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel(M,kernelType=:expiSym));
KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantKernel_Gaussian(M));
KP = FieldDependentKernelProblem(M);
LK = MLKernel.LearnKernel(KP,:driftOpt,10; tspan=40.,NTr=30,
                            #tspan_test=20.,NTr_test=20,
                            saveat=0.05,
                          #Ys=[1.0,1.5,2.,3.,4.],
                          #opt=MLKernel.Nesterov(),
                          opt=MLKernel.ADAM(0.001),
                          runs_pr_epoch=10);#, Ys=[1.,1.5,2.,4.,5.])
learnKernel(LK; cb=cb)

LK.KP = MLKernel.updateProblem(KP);

cb(LK)


@run cb(LK)


lhistory_H_RWNoise = lhistory
lhistory_H_RWNoNoise = lhistory
lhistory_expiP_RWNoNoise = lhistory
lhistory_expiP_RWNoise = lhistory
lhistory_H = lhistory
lhistory_expiP = lhistory

lhistory_K = lhistory
lhistory_expiP = lhistory

lhistory_expiP_6 = lhistory
lhistory_expiP_10 = lhistory
lhistory_expiP_14 = lhistory

lhistory_expiP_DimDre = lhistory_expiP_DreUim# = lhistory
lhistory_expiP_Dim = lhistory
lhistory_expiPHerm = lhistory
lhistory_expiPSym = lhistory
lhistory_expAiP = lhistory

begin
    fig = plot(xlabel="Iteration",yaxis=:log)#,ylim=[1e-3,1e3])
    plot!(fig,lhistory[:LTrue],label="LTrue")
    plot!(fig,lhistory[:L],label="LTrain")
    plot!(fig,lhistory[:LSymDrift],label="LSymDrift")
end

begin
    fig = plot(xlabel="Iteration",yaxis=:log)#,ylim=[1,1e6])
    #plot!(fig,lhistory[:L],color=3)
    #plot!(fig,lhistory_K[:LTrue],label=L"LTrue $H$")
    #plot!(fig,lhistory_K[:L],label=L"LTrain $H$")
    plot!(fig,lhistory_expiP[:LTrue],label=L"LTrue $e^{iP}$")
    plot!(fig,lhistory_expiP[:L],label=L"LTrain $e^{iP}$")
    plot!(fig,lhistory_expiPHerm[:LTrue],label=L"LTrue $e^{iP_{\textrm{Herm}}}$")
    plot!(fig,lhistory_expiPHerm[:L],label=L"LTrain $e^{iP_{\textrm{Herm}}}$")
    plot!(fig,lhistory_expiPSym[:LTrue],label=L"LTrue $e^{iP_{\textrm{Sym}}}$")
    plot!(fig,lhistory_expiPSym[:L],label=L"LTrain $e^{iP_{\textrm{Sym}}}$")
end

begin
    fig = plot(xlabel="Iteration",yaxis=:log)#,ylim=[1,1e6])
    #plot!(fig,lhistory[:L],color=3)
    #plot!(fig,lhistory_K[:LTrue],label=L"LTrue $H$")
    #plot!(fig,lhistory_K[:L],label=L"LTrain $H$")
    #plot!(fig,lhistory_H[:LTrue],label="LTrue imDrift",color=1)
    #plot!(fig,lhistory_H[:L],label="LDrift imDrift",color=1,linestyle=:dash)
    #plot!(fig,lhistory_H[:LSymDrift],label="LSymD imDrift",color=1,linestyle=:dashdot)
    plot!(fig,lhistory_expiP[:LTrue],label="LTrue imDrift_expiP",color=2)
    plot!(fig,lhistory_expiP[:L],label="LDrift imDrift_expiP",color=2,linestyle=:dash)
    plot!(fig,lhistory_expiP[:LSymDrift],label="LSymD imDrift_expiP",color=2,linestyle=:dashdot)
    #plot!(fig,lhistory_H_RWNoise[:LTrue],label="LTrue LSymApprox",color=3)
    #plot!(fig,lhistory_H_RWNoise[:L],label="LDrift LSymApprox",color=3,linestyle=:dash)
    #plot!(fig,lhistory_H_RWNoise[:LSymDrift],label="LSymD LSymApprox",color=3,linestyle=:dashdot)
    #plot!(fig,lhistory_H_RWNoNoise[:LTrue],label="LTrue LSymApproxNoNoise",color=4)
    #plot!(fig,lhistory_H_RWNoNoise[:L],label="LDrift LSymApproxNoNoise",color=4,linestyle=:dash)
    #plot!(fig,lhistory_H_RWNoNoise[:LSymDrift],label="LSymD LSymApproxNoNoise",color=4,linestyle=:dashdot)
    #plot!(fig,lhistory_expiP_RWNoNoise[:LTrue],label="LTrue LSymApproxNoNoise_expiP",color=5)
    #plot!(fig,lhistory_expiP_RWNoNoise[:L],label="LDrift LSymApproxNoNoise_expiP",color=5,linestyle=:dash)
    #plot!(fig,lhistory_expiP_RWNoNoise[:LSymDrift],label="LSymD LSymApproxNoNoise_expiP",color=5,linestyle=:dashdot)
    plot!(fig,lhistory_expiP_RWNoise[:LTrue],label="LTrue LSymApprox_expiP",color=6)
    plot!(fig,lhistory_expiP_RWNoise[:L],label="LDrift LSymApprox_expiP",color=6,linestyle=:dash)
    plot!(fig,lhistory_expiP_RWNoise[:LSymDrift],label="LSymD LSymApprox_expiP",color=6,linestyle=:dashdot)
end

begin
    fig = plot(xlabel="Iteration")#,ylim=[1,1e6])
    #plot!(fig,lhistory[:L],color=3)
    #plot!(fig,real(lhistory_K[:detK]),label=L"Re|K| $H$")
    #plot!(fig,imag(lhistory_K[:detK]),label=L"Im|K| $H$")
    plot!(fig,real(lhistory_expiP[:detK]),label=L"Re|K| $e^{iP}$")
    plot!(fig,imag(lhistory_expiP[:detK]),label=L"Im|K| $e^{iP}$")
    plot!(fig,real(lhistory_expAiP[:detK]),label=L"Re|K| $e^{A+iP}$")
    plot!(fig,imag(lhistory_expAiP[:detK]),label=L"Im|K| $e^{A+iP}$")
end







_p = MLKernel.getKernelParams(LK.KP.kernel);
det(_p[1:div(end,2),:] + im*_p[div(end,2)+1:end,:])
2*LK.KP.model.contour.t_steps*abs(det(_p[1:div(end,2),:] + im*_p[div(end,2)+1:end,:]) - 1)
2.0*LK.KP.model.contour.t_steps*100*abs(det(_p[1:div(end,2),:] + im*_p[div(end,2)+1:end,:]) - 1)
100*abs(det(_p[div(end,2)+1:end,:]) - 1)
abs(det(LK.KP.kernel.pK.sqrtK[1:div(end,2),:] + im*LK.KP.kernel.pK.sqrtK[div(end,2)+1:end,:]) - 1)



_KReal = LK.KP.kernel.pK.K
_HComplex = LK.KP.kernel.pK.sqrtK[1:div(end,2),:] + im*LK.KP.kernel.pK.sqrtK[div(end,2)+1:end,:]
_KComplex = LK.KP.kernel.pK.K[1:div(end,2),1:div(end,2)] + im*LK.KP.kernel.pK.K[div(end,2)+1:end,1:div(end,2)]
spy(_KReal;markersize=5)
det(_KReal)
det(_KComplex)
det(_HComplex)



_K2Real = KP2.kernel.pK.K
_H2Complex = KP2.kernel.pK.sqrtK[1:div(end,2),:] + im*KP2.kernel.pK.sqrtK[div(end,2)+1:end,:]
_K2Complex = KP2.kernel.pK.K[1:div(end,2),1:div(end,2)] + im*KP2.kernel.pK.K[div(end,2)+1:end,1:div(end,2)]
spy([real(_H2Complex);imag(_H2Complex)],markersize=2.5)
det(_K2Real)
det(_K2Complex)
det(_H2Complex)

spy(real(_H2Complex*inv(_H2Complex));markersize=7)
spy(imag(_H2Complex*inv(_H2Complex));markersize=7)
spy(imag(_HComplex*adjoint(_HComplex));markersize=7)

evalsKReal = eigvals(_KReal)
evalsKComplex = eigvals(_KComplex)
evalsHComplex = eigvals(_HComplex)
scatter(evalsKReal)
scatter(evalsKComplex)
scatter(evalsHComplex)
minimum(real(evalsKComplex))
maximum(real(evalsKComplex))
minimum(imag(evalsKComplex))

evalsK2Real = eigvals(_K2Real)
evalsK2Complex = eigvals(_K2Complex)
evalsH2Complex = eigvals(_H2Complex)
evalsH2Complex,evecsH2Complex = eigen(_H2Complex)
scatter(evalsK2Real,xlim=[-0.01,0.5])
scatter(evalsK2Complex)#,xlim=[-0.01,0.5])
scatter(evalsH2Complex,xlim=[-0.01,1.])
minimum(real(evalsH2Complex))
minimum(imag(evalsH2Complex))

sort(evalsHComplex; by=real)#[end-10:end]
diff(imag(sort(evalsH2Complex; by=real)))#[end-10:end]

sum(imag(evalsHComplex))
sum(imag(evalsH2Complex))

spy(imag(_H2Complex-transpose(_H2Complex))) #- transpose(_K2Complex)*_K2Complex

real(_H2Complex-adjoint(_H2Complex))
imag(_H2Complex+adjoint(_H2Complex))

begin
S = 0.
for evi in evalsHComplex
    mindiff = 1000.
    for evj in evalsHComplex
        d = abs(evi - conj(evj))
        if d < mindiff
            mindiff = d
        end
    end
    S += mindiff
end
    S
end


θ = _p[1:div(end,2),:]
s = _p[div(end,2)+1:end,:]
HRe = s .* cos.(θ)
HIm = s .* sin.(θ)


begin
_p = MLKernel.getKernelParams(KP.kernel);
KK = _p[1:div(end,2),:] + im*_p[div(end,2)+1:end,:]
KK2 = KK^2
invKK2 = inv(KK2)
invKK2Re = real(invKK2)
invKK2Im = imag(invKK2)
invKK2Re[abs.(invKK2Re) .< 0.01] .= 0
invKK2Im[abs.(invKK2Im) .< 0.01] .= 0
sp_invKK2Re = sparse(invKK2Re)
sp_invKK2Im = sparse(invKK2Im)
eye = [Diagonal(diag(sp_invKK2Re));Diagonal(diag(sp_invKK2Im))]
spy([sp_invKK2Re;sp_invKK2Im] .- eye, markersize=4)
end

BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
@time B1,B2 = calcBoundaryTerms(sol,BT);

begin
fig = plot(
    #title=MLKernel.L"$\textrm{Re}\langle L_c\;x(0)x(t) \rangle$",
    title=MLKernel.L"$\textrm{Re}\langle L_c\;x^2 \rangle$",
    xlabel=MLKernel.L"$Y$",ylabel=MLKernel.L"B_1(Y)";
    MLKernel.plot_setup(false)...)
fig2 = plot(
    #title=MLKernel.L"$\textrm{Re}\langle L_c\;x(0)x(t) \rangle$",
    title=MLKernel.L"$\textrm{Re}\langle L_c\;x^2 \rangle$",
    xlabel=MLKernel.L"$Y$",ylabel=MLKernel.L"B_1(Y)";
        MLKernel.plot_setup((1.25, 1.0))...)
for i in [1,3,6,10,11,12]#10:12#KP.model.contour.t_steps
    t_p = round(KP.model.contour.tp[i],digits=1)
    #plot!(fig,Ys,B1[0*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
    #plot!(fig2,Ys,B1[1*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
    #plot!(fig,Ys,B1[2*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
    #plot!(fig2,Ys,B1[3*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
    plot!(fig,Ys,B1[4*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
    plot!(fig2,Ys,B1[5*KP.model.contour.t_steps+i,:],label=MLKernel.L"$t_p = %$t_p$";MLKernel.markers_dict(i)...)
end
plot(fig,fig2,layout=grid(1, 2, widths=[0.4, 0.4]),size=(900,300))
end

begin
    i = 21
    k = 4
    _B = copy(B1[k*KP.model.contour.t_steps+i:k*KP.model.contour.t_steps+i+1,:,:])
    _B[2,:,:] = B1[(k+1)*KP.model.contour.t_steps+i,:,:]
    plotB(_B,BT;plotType=:surfaceAndLines)
end








###
filename = "HO_10_freeKernel.jld"
@save filename :sol=sol :KP=KP :l=l
lo = load(filename)



###
p0 = [1.8,0.8]
M=AHO(1.,24.,1.5,1.,10)
function L(p;plotL=false)
    KP = ConstantKernelProblem(M;kernel=MLKernel.ConstantFreeKernel(M;m=p[1],g=p[2]))
    l, sol = trueLoss(KP,tspan=20,NTr=50);
    if plotL
        display(MLKernel.plotSKContour(KP,sol))
    end
    l
end


using ForwardDiff
using Zygote
using FiniteDifferences
using Flux

@time L([1.8,0.8],plotL=true)
#@time Zygote.gradient(L,[1.,1.])

@time ForwardDiff.gradient(L,[1.,1.])
#@time FiniteDifferences.grad(forward_fdm(2,1),L,[1.,1.])

ForwardDiff.jacobian(p -> MLKernel.ConstantFreeKernel(M; m=p[1], g=p[2]).pK.sqrtK,[1.,1.])
Zygote.jacobian(p -> MLKernel.ConstantFreeKernel(M; m=p[1], g=p[2]).pK.sqrtK,[1.,1.])[1]


begin
#p = copy(p0)
opt = ADAM(0.2)
for i in 1:10
    @time dp = ForwardDiff.gradient(L,p)
    #dp = FiniteDifferences.grad(forward_fdm(2,1),L,p)[1]
    Flux.update!(opt,p,dp)
    @show p,dp,L(p;plotL=true)
end
end







g(x::Dual{Z,T,N}) where {Z,T,N} = Dual{Z}(NaN, ntuple(i -> 5 * partials(x,i), N))


_g(x::Vector) = [x[1]*x[2],x[2]]
_g([1.,2.])

jacobian(x -> _g(x), [1,2])  # previously an error
g(x::Dual{Z,T,N}) where {Z,T,N} = Dual{Z}(NaN, ntuple(i -> 5 * partials(x,i), N))

g(x::Real) = x
g(x::Dual{Z,T,N}) where {Z,T,N} = Dual{Z}(value(x), (partials(x,1), partials(x,2)))

g(x::Vector) = [x[1]*x[2],x[1]]
_g(x::Vector) = [x[1]*x[2],x[1]]
g(x::Vector{Dual{Z,T,N}}) where {Z,T,N} = begin 
    [
        Dual{Z}(value(x[1]*x[2]), (value(x[2])*partials(x[1],1), value(x[1])*partials(x[1],2))),
        Dual{Z}(value(x[2]), ( partials(x[2],1), partials(x[2],2)))
    ] 
end

g([1,2])

jacobian(x -> _g(x), [1,2])  # previously an error
jacobian(x -> g(x), [1,2])  # previously an error


B(p) = p[1]*Matrix(Diagonal(randn(1000)))


F(p) = begin
    inv(p)
    vals, vecs = eigen(p)
    return real(vecs * Diagonal(sqrt.(vals .+ 0*im)) * inv(vecs))
end

function F(p::Matrix{ForwardDiff.Dual{Z,T,N}}) where {Z,T,N}
    #@show typeof(p)
    @show size(p)
    @show typeof(ForwardDiff.value.(p))
    dsqrt = Zygote.jacobian(F,ForwardDiff.value.(p))[1]
    @show size(dsqrt)
    @show typeof([ForwardDiff.Dual{Z}(1.,ntuple(k -> ForwardDiff.partials(p[i,j]),N)) for i in 1:3, j in 1:3])#ntuple(i -> dsqrt[i]*partials(p,i), N)) 
    @show typeof(inv(p))
    @show size(inv(p))
    return inv(p)
end

function g(p::Matrix)
    M = inv(p)
    return F(M)
end 



A = randn(3,3)
F(A)
Zygote.jacobian(F,A)[1]


p = randn(3,3)
g(p)
Zygote.jacobian(g,p)[1]
ForwardDiff.jacobian(g,A)

Zygote.jacobian(p -> p*p,[1 0 0 ; 0 2 0 ; 0 0 3])[1]



_F(p) = 2*sqrt(p)
function _F(p::Dual{Z,T,N}) where {Z,T,N} 
    dsqrt = Zygote.gradient(_F,value(p))[1]
    Dual{Z}(_F(value(p)),ntuple(i -> dsqrt*partials(p,i), N)) 
end
_g(p) = _F(p^2)

Zygote.gradient(_g,3.)[1]
ForwardDiff.derivative(_g,3.)


using ForwardDiff
goo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
foo((x, y, z),) = [x^2*z, x*y*z, abs(z)-y]
function foo(u::Vector{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # get f(vs)
    val = foo(vs)
    # get J(f, vs) * ps (cheating). Write your custom rule here
    jvp = Zygote.jacobian(goo, vs)[1] * ps
    # pack: SoA -> AoS
    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end

ForwardDiff.gradient(u->sum(cumsum(foo(u))), [1, 2, 3]) == 
ForwardDiff.gradient(u->sum(cumsum(goo(u))), [1, 2, 3])


goo(M) = inv(M)
foo(M) = inv(M)
function foo(u::Matrix{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # get f(vs)
    val = foo(vs)
    # get J(f, vs) * ps (cheating). Write your custom rule here
    jvp = Zygote.jacobian(foo, vs)[1] * ps
    # pack: SoA -> AoS
    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end

M = randn(2,2)
Zygote.jacobian(u->foo(u*u), M)[1]# == 
ForwardDiff.jacobian(u->goo(u*u), M)


F(p) = begin
    inv(p)
    vals, vecs = eigen(p)
    return real(vecs * Diagonal(sqrt.(vals .+ 0*im)) * inv(vecs))
end

function F(u::Matrix{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    # you can play with the dimension here, sometimes it makes sense to transpose
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    # get f(vs)
    val = foo(vs)
    # get J(f, vs) * ps (cheating). Write your custom rule here
    jvp = Zygote.jacobian(F, vs)[1] * ps
    # pack: SoA -> AoS
    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end

M = randn(2,2)
Zygote.jacobian(u -> F(u*u), M)[1]
ForwardDiff.jacobian(u -> F(u*u), M)


ff(p) = begin
    inv(p)
    vals, vecs = eigen(p)
    return real(vecs * Diagonal(sqrt.(vals .+ 0*im)) * inv(vecs))
end
function ff(u::Matrix{ForwardDiff.Dual{T,V,P}}) where {T,V,P}
    # unpack: AoS -> SoA
    vs = ForwardDiff.value.(u)
    ps = mapreduce(ForwardDiff.partials, hcat, u)
    val = ff(vs)

    J = Zygote.jacobian(Y -> ff(Y),vs)[1]
    jvp = J * ps'

    return map(val, eachrow(jvp)) do v, p
        ForwardDiff.Dual{T}(v, p...) # T is the tag
    end
end

M = randn(2,2)
Zygote.jacobian(u->ff(u*u), M)[1]# == 
ForwardDiff.jacobian(u->ff(u*u), M)

using ForwardDiff: Dual, value, partials

function _eigvals(A::Symmetric{<:ForwardDiff.Dual{Tg,T,N}}) where {Tg,T<:Real,N}
    
    ps = mapreduce(ForwardDiff.partials, hcat, A)

    # get J(f, vs) * ps (cheating). Write your custom rule here
    J = Zygote.jacobian(Y -> eigvals(Symmetric(Y)),ForwardDiff.value.(A))[1]
    @show J
    @show ps
    jvp = J * ps'
    @show jvp    

    λ,Q = eigen(Symmetric(ForwardDiff.value.(parent(A))))
    parts = ntuple(j -> diag(Q' * getindex.(ForwardDiff.partials.(A), j) * Q), N)
    @show parts
    ForwardDiff.Dual{Tg}.(λ, tuple.(parts...))
end


M
MM = real(sqrt(Symmetric(M)))
parent(MM)
issymmetric(MM*MM)
ForwardDiff.gradient(u->sum(real(_eigvals(Symmetric(u*u)))), M)
Zygote.gradient(u->sum(real(eigvals(Symmetric(u*u)))), M)[1]
ForwardDiff.jacobian(u->real(_eigvals(Symmetric(u*u))), M)
Zygote.jacobian(u->real(eigvals(Symmetric(u*u))), M)[1]





v, back = Zygote.pullback(J -> inv(J), [1 0 ; 0 1])
Zygote.jacobian(J -> inv(J), [1 0 ; 0 1])[1]

back([1 0 ; 0 0])[1]
back([0 0 ; 0 1])[1]

_vec(x) = x
 _vec(x::AbstractArray) = vec(x)

jacobians = map(i -> zeros(Float64,length(v),i),[1 0 ; 0 1])

for (k, idx) in enumerate(eachindex(v))
    @show k 
    grad_output = fill!(v, 0)
    grad_output[idx] = 1
    grads_input = back(grad_output)
#    for (jacobian_x, d_x) in zip(jacobians, grads_input)
#        @show _vec(d_x)
#        @show jacobian_x[k, :]
#        jacobian_x[k, :] .= _vec(d_x)
#    end
end

back([1 0 ; 0 0])
back([0 1 ; 0 0])
back([0 0 ; 1 0])
back([0 0 ; 0 1])





p_expiPHerm = LK.KP.kernel.pK.p


begin
#_κR = exp.( -(real(M.contour.x0) .- transpose(real(M.contour.x0))).^2/0.8) 
#_κI = exp.( -(imag(M.contour.x0) .- transpose(imag(M.contour.x0))).^2/0.8) 
#_κRI = exp.( -(real(M.contour.x0) .- transpose(imag(M.contour.x0))).^2/0.8) 
#_κIR = exp.( -(imag(M.contour.x0) .- transpose(real(M.contour.x0))).^2/0.8) 
#_M = 0.5*(_κR + _κI + im*(_κRI - _κIR))

#_M = exp.( im*transpose(M.contour.x0 .- adjoint(M.contour.x0)) * ((M.contour.x0 .- adjoint(M.contour.x0)))/1)
#_M = exp( conj(M.contour.x0 .- transpose(M.contour.x0)) .* (M.contour.x0 .- transpose(M.contour.x0))/0.8)
#_M = _κR

γ = 0.8
#vr=1
#vrj = 1
dx = norm.(M.contour.x0 .- transpose(M.contour.x0))
#μ = ones(ComplexF64,length(M.contour.x0))
#_M = (vr^2+vrj^2)*exp.( -conj.(dx) .* (dx)/γ) +
#     im*vr*vrj*(exp.( -conj.(dx .- μ) .* (dx .- μ)/γ) -
#     exp.(-conj.(dx .+ μ) .* (dx .+ μ)/γ))
#p=3*pi/2
#_M = 1^2*exp.(-sin.(π*abs.(dx)/p))*
_M = exp.(-real(dx).^2/2)


spy([real(_M) ; imag(_M)],markersize=6)
end

begin
x = collect(-5:0.1:5)    
y = collect(-5:0.1:5)    

γ = 0.8
vr=1
vrj = 1
dx = x .+ transpose(im*y)
μ = zeros(ComplexF64,length(x))
_M = (vr^2+vrj^2)*exp.( -conj.(dx) .* (dx)/γ) +
     im*vr*vrj*(exp.( -conj.(dx .- μ) .* (dx .- μ)/γ) -
     exp.(-conj.(dx .+ μ) .* (dx .+ μ)/γ))
#_M = exp.( im*transpose(M.contour.x0 .- adjoint(M.contour.x0)) * ((M.contour.x0 .- adjoint(M.contour.x0)))/1)
#_M = 0.5*(_κR + _κI + im*(_κRI - _κIR))
spy([real(_M) ; imag(_M)],markersize=3)
end