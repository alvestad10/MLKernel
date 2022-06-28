using MLKernel
using Plots
using Measurements
using LinearAlgebra, Statistics
using JLD2
using SparseArrays
using LaTeXStrings

M=AHO(1.,0.,10.,1.0,10)


KP_noKernel = ConstantKernelProblem(M)
KP_freeKernel = ConstantKernelProblem(M;kernel=MLKernel.ConstantFreeKernel(M;m=1.,g=1.0));


@time l_noKernel , sol_noKernel = trueLoss(KP_noKernel,tspan=100.,NTr=50,saveat=0.01);
@time l_freeKernel , sol_freeKernel = trueLoss(KP_freeKernel,tspan=100.,NTr=50,saveat=0.01);



### SAVING AND LO
const RESULT_DIR = "Results/FreeTheory/"

path_noKernel = joinpath(RESULT_DIR,"configs_noKernel.jld2")
path_freeKernel = joinpath(RESULT_DIR,"configs_freeKernel.jld2") 


@save path_noKernel :sol=sol_noKernel :KP=KP_noKernel :l=l_noKernel
@save path_freeKernel :sol=sol_freeKernel :KP=KP_freeKernel :l=l_freeKernel

begin
    lo_noKernel = load(path_noKernel)
    sol_noKernel = lo_noKernel[":sol"]
    KP_noKernel = lo_noKernel[":KP"]
    l_noKernel = lo_noKernel[":l"]
end

begin
    lo_freeKernel = load(path_freeKernel)
    sol_freeKernel = lo_freeKernel[":sol"]
    KP_freeKernel = lo_freeKernel[":KP"]
    l_freeKernel = lo_freeKernel[":l"]
end


##### PLOTTING
fig_noKernel = MLKernel.plotSKContour(KP_noKernel,sol_noKernel)
fig_freeKernel = MLKernel.plotSKContour(KP_freeKernel,sol_freeKernel)

savefig(fig_noKernel,joinpath(RESULT_DIR,"RT_Observables_FreeTheory_noKernel.pdf"))
savefig(fig_freeKernel,joinpath(RESULT_DIR,"RT_Observables_FreeTheory_freeKernel.pdf"))
