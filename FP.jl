using MLKernel
using Measurements
using LinearAlgebra, Statistics
using LaTeXStrings
using KrylovKit
using BlockBandedMatrices, BandedMatrices, SparseArrays, BenchmarkTools, LinearAlgebra
using FillArrays, LazyBandedMatrices, LazyArrays
using Plots


function getMatrix(KP::MLKernel.KernelProblem)
    ∇S(u) = begin
        du = similar(u)
        KP.a(du,u,[1.,0.],0.)
        return du
    end

    σ = KP.model.σ
    λ = KP.model.λ
    _σ = σ[1] + im*σ[2]
    Hfp(u) = begin
        return -(_σ^2 / 4)*u^2 + (3*(λ/6)/2)*u^2 - (_σ*(λ/6)/2)u^4 - ((λ/6)^2/4)*u^6 + (_σ/2) 
    end



    n = 500
    xmin = -6; xmax = 6
    x = range(xmin,xmax;length=n)
    y = range(xmin,xmax;length=n)
    dx = x[2]-x[1]

    
    #=Is = [1:n-1; 2:n]; Js = [2:n; 1:n-1]; Vs = [fill(1, n-1);fill(-1, n-1)];
    D = sparse(Is, Js, Vs) ./ (dx)^2
    #D[1,n] = -1 ./ (dx)^2
    #D[n,1] = -1 ./ (dx)^2
    Dx = kron(D, sparse(I,n,n));
    Dy = kron(sparse(I,n,n), D);

    Sr = zeros(n^2)
    Si = zeros(n^2)
    for K in 1:n
        dS = hcat([∇S([xi,y[K]]) for xi in x]...)
        Sr[n*(K-1)+1:n*K] = dS[1,:]
        Si[n*(K-1)+1:n*K] = dS[2,:]
    end
    Sr = Diagonal(Sr)
    Si = Diagonal(Si)=#

    Sr = BandedBlockBandedMatrix(Zeros(n^2,n^2), Fill(n,n),Fill(n,n), (0,0), (0,0))
    Si = BandedBlockBandedMatrix(Zeros(n^2,n^2), Fill(n,n),Fill(n,n), (0,0), (0,0))
    Dx = BandedBlockBandedMatrix(Zeros(n^2,n^2), Fill(n,n),Fill(n,n), (0,0), (1,1))
    Dy = BandedBlockBandedMatrix(Zeros(n^2,n^2), Fill(n,n),Fill(n,n), (1,1), (0,0))
    for K = 1:n
        dS = hcat([∇S([xi,y[K]]) for xi in x]...)
        view(Sr,Block(K,K))[band(0)] .= dS[1,:]; 
        view(Si,Block(K,K))[band(0)] .= dS[2,:]; 
        
        view(Dx,Block(K,K))[band(1)] .= 1. /(2dx); 
        view(Dx,Block(K,K))[band(-1)] .= -1. /(2dx);

        if K<n
            view(Dy,Block(K,K+1))[band(0)] .= 1. /(2dx); 
            view(Dy,Block(K+1,K))[band(0)] .= -1. /(2dx);
        end
    end
    
    A = -(Dx*(Dx - Sr) + Dy*(-Si))
    #KRe, KIm = KP.kernel.K([],MLKernel.getKernelParams(KP.kernel))
    #A = -KRe*(Dx*(Dx - Sr) + Dy*(-Si)) + KIm*(Dy*(Dx - Sr) - Dx*(-Si))
    #A = -(KRe^2*Dx*Dx + KIm*Dy*Dy + 2*KRe*KIm*Dx*Dy - Dx*(KRe*Sr-KIm*Si) - Dy*(KRe*Si + KIm*Sr))


    #A = transpose(vec([Dx,Dy]))*vec(K*vec([Dx + Sr,Si])) 

    


    #=n = 30000
    xmin = -15; xmax = 15
    x = collect(range(xmin,xmax;length=n))
    dx = x[2]-x[1]
    dS = BandedMatrix{ComplexF64}(undef,(n,n), (0,0))
    D2x = BandedMatrix{Float64}(undef,(n,n), (1,1))
    
    #@show size(dS[band(0)])
    
    #dS = hcat([∇S([xi,0.]) for xi in x]...)
    #@show size(dS[1,:] .+ im*dS[2,:])

    Hfp = Hfp.(x)
    dS[band(0)] .= Hfp; 
    D2x[band(0)] .= -2. /(dx)^2; 
    D2x[band(1)] .= 1. /(dx)^2; 
    D2x[band(-1)] .= 1. /(dx)^2;
    D2x[band(n)] .= 1. /(dx)^2; 
    D2x[band(-n)] .= 1. /(dx)^2;=#

    #KRe, KIm = KP.kernel.K([],MLKernel.getKernelParams(KP.kernel))

    #A = -(KRe + im*KIm) .* (0.5*D2x + dS)
    return A
end

M = LM_AHO([0.,4],12.)
KP = ConstantKernelProblem(M);
A = getMatrix(KP)

function test(KP::MLKernel.KernelProblem)

    A = getMatrix(KP)

    #vals,vecs,info = KrylovKit.eigsolve(A,3,:SR)#,rand(size(A,1)),1,EigSorter(abs; rev = false),Arnoldi())
    @time vals,vecs,info = KrylovKit.eigsolve(sparse(A),5,EigSorter(abs; rev = false))#,rand(size(A,1)),1,EigSorter(abs; rev = false),Arnoldi())
    #@time vals2,_ = eigs(A; nev=2, which=:SR)
    #vals = eigvals(Matrix(A))#;sortby=(λ) -> real(λ))
    #decomp, history = partialschur(A, nev=3, tol=1e-1, which=SR(), restarts=20000);
    #vals = decomp.eigenvalues
    #@show history
    return vals

    
end


begin
Bs = collect(0:0.5:1)
vals1 = []
vals2 = []
vals3 = []
all_ls = []
for i in Bs
    M = LM_AHO([0.,i],12.)
    KP = ConstantKernelProblem(M);
    
    vals = test(KP)
    append!(vals1,vals[1])
    append!(vals2,vals[2])
    append!(vals3,vals[3])
    
    @time l,_ = trueLoss(KP,tspan=10,NTr=10);
    append!(all_ls,l)

    println("B=",i, ", L=", round(l,digits=5), ", λ₁=",vals[1], ", λ₂=",vals[2], ", λ₃=",vals[3])
end
plot(Bs,real(vals1))
#plot!(Bs,real(vals2))
#plot!(Bs,real(vals3))
plot!(Bs,all_ls)
end
 

Bs_B = Bs 
vals1_B = vals1
ls_B = all_ls
begin
plot(Bs_B,real(vals1_B))
plot!(Bs_B,ls_B)
end


begin
    θs = [0.]#collect(-pi:0.1:0)
    vals1 = []
    all_ls = []
    for θ in θs
        M = LM_AHO([0.,4],12.)
        KP = ConstantKernelProblem(M);
        setKernel!(KP.kernel,
            [θ]
        )

        vals = test(KP)
        append!(vals1,vals[1])
        
        @time l,_ = trueLoss(KP,tspan=100,NTr=100);
        append!(all_ls,l)
    
        println("θ=",θ, ", L=", round(l,digits=5), ", λ₁=",vals[1], ", λ₂=",vals[2], ", λ₃=",vals[3])
    end
    plot(θs,real(vals1),ylim=[-0.1,0.1])
    plot!(θs,all_ls)
end

M = LM_AHO([0.,4],12.)
KP = ConstantKernelProblem(M);
MLKernel.calcLFP(KP)

A = [1 2 ; 3 4]
B = [2 3 ; 4 5]
C = similar(A)


function f(x)
    x^2
end

fff(x) = x^2



du = [1,0,0,0,0,0]
f!(du,u) = (du .= u.^2)

a = f!


a(du,[1,2,3,4])

du





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
    #bt =  MLKernel.calcBTLoss(sol,KP,LK.Loss.Ys)
    #rt =  MLKernel.calcRealPosDrift(sol,KP)
    #LSymDrift =  MLKernel.calcLSymDrift(sol,KP)
    #reD =  MLKernel.calcReDrift(sol,KP)
    #imD =  MLKernel.calcImDrift(sol,KP)
    #LSym =  MLKernel.calcSymLoss(sol,KP)
    LFP =  MLKernel.calcLFP(KP)


    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, avg3Re, err_avg3Re, avg3Im, err_avg3Im = MLKernel.calc_meanObs(KP;sol=sol)
    @show KP.y["x2"],(avg2Re .± err_avg2Re)[1], (avg2Im .± err_avg2Im)[1]
    @show (avgRe .± err_avgRe)[1], (avgIm .± err_avgIm)[1], (avg3Re .± err_avg3Re)[1], (avg3Im .± err_avg3Im)[1]

    if KP.kernel isa MLKernel.ConstantKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", LFP: ", LFP ,", Ks: ", MLKernel.getKernelParams(KP.kernel))
    elseif KP.kernel isa MLKernel.FunctionalKernel || KP.kernel.pK isa MLKernel.LM_AHO_NNKernel || KP.kernel.pK isa MLKernel.LM_AHO_FieldKernel
        println("LTrain: ", LTrain, ", TLoss: ", TLoss, ", LFP: ", LFP )
    end

    #=if addtohistory
        append!(lhistory[:L],LTrain)
        append!(lhistory[:LTrue],TLoss)
        append!(lhistory[:imDrift],imD)
        append!(lhistory[:LSymDrift],LSymDrift)
        append!(lhistory[:BT],bt)
        append!(lhistory[:RT],rt)
        if KP.kernel isa MLKernel.AHO_ConstKernel
            append!(lhistory[:Ks],[KP.kernel.pK.K])
        end
    end=#

    #if LK.Loss.ID ∈ [:BT, :BTSym, :driftOpt]
    #    BT = getBoundaryTerms(KP;Ys=Ys)#,Xs=Xs);
    #    B1,B2 = calcBoundaryTerms(sol,BT;witherror=true);
    #    display(plotB(B1,BT))
    #end

end

M = LM_AHO([-1.,3],12.)
KP = ConstantKernelProblem(M);
LK = MLKernel.LearnKernel(KP,:FP,10; tspan=50.,NTr=50,saveat=0.01,
                          #Ys=[0.6,1.0,1.5,2.,3.,4.],
                          opt=MLKernel.ADAM(0.05),
                          runs_pr_epoch=1);
learnKernel(LK; cb=cb)








begin
M = LM_AHO([0.,2],12.)

NH = 50
NI = 0.
NR = 1. + NI
ω = 1

A = M.σ[1] / ω 
B = M.σ[2] / ω 
λ = (M.λ/6) / ω 

H = spzeros(NH^2,NH^2)

f(a,b) = Float64(sqrt(factorial(big(a)) / factorial(big(b))))

X(k,l,m,n) = begin
    v = 0.
    if (l == n && k == m+4)  ; v += f(k,m)            ; end
    if (l == n && k == m+2)  ; v += (2m+3-6n)f(k,m)   ; end
    if (l == n && k == m)    ; v += 6*(m-n)           ; end
    if (l == n && k == m-2)  ; v += -(2m-7-6n)*f(m,k) ; end
    if (l == n && k == m-4)  ; v += -f(m,k)           ; end
    if (l == n+2 && k == m+2); v += -3*f(k,m)*f(l,n)  ; end
    if (l == n-2 && k == m+2); v += -3*f(k,m)*f(n,l)  ; end
    if (l == n+2 && k == m-2); v +=  3*f(m,k)*f(l,n)  ; end
    if (l == n-2 && k == m-2); v +=  3*f(m,k)*f(n,l)  ; end
    if (l == n+2 && k == m)  ; v +=  -3*f(l,n)        ; end
    if (l == n-2 && k == m)  ; v += -3*f(n,l)         ; end
    return v
end

#for i in 1:floor(Integer,NH^2)
#    for j in 1:floor(Integer,NH^2)
        #k = div(i-1 - mod(i-1,NH),NH)
        #l = mod(i-1,NH)
        #m = div(j-1 - mod(j-1,NH),NH)
        #n = mod1(j-1,NH)
for k in 0:NH-1
for l in 0:NH-1
for m in 0:NH-1
for n in 0:NH-1
    i = k*NH + l + 1
    j = m*NH + n + 1

        if (k == m && l == n)     ;  H[i,j] = (NR-λ)*(2m+1) + (NI + λ)*(2n+1) - 2A ; end
        if (k == m+2 && l == n)   ;  H[i,j] = -(NR+λ-A)*f(k,m)                    ; end
        if (k == m-2 && l == n)   ;  H[i,j] = -(NR+λ-A)*f(m,k)                    ; end
        if (k == m && l == n+2)   ;  H[i,j] = -(NI-λ-A)*f(l,n)                    ; end
        if (k == m && l == n-2)   ;  H[i,j] = -(NI-λ+A)*f(n,l)                    ; end
        if (k == m-1 && l == n+1) ;  H[i,j] = 2B*sqrt(m*l)                        ; end
        if (k == m+1 && l == n-1) ;  H[i,j] = -2B*sqrt(k*n)                       ; end

        
        X_klmn = X(k,l,m,n)
        if X_klmn != 0.
            H[i,j] += (λ/12)*X_klmn
        end

        X_lknm = X(l,k,n,m)
        if X_lknm != 0.
            H[i,j] += -(λ/12)*X_lknm
        end 
end
end
end
end
vals,vecs,info = KrylovKit.eigsolve(H*(2/ω),10,
                #EigSorter(abs; rev = false))
                :SR)
display(scatter(vals))
vals
end