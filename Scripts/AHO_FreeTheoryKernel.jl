using MLKernel
using Plots
using Measurements
using LinearAlgebra, Statistics
using JLD2
using SparseArrays
using LaTeXStrings

M=AHO(1.,24.,1.,1.0,20)


KP_noKernel = ConstantKernelProblem(M)
KP_freeKernel = ConstantKernelProblem(M;kernel=MLKernel.ConstantFreeKernel(M;m=1.,g=1.0));
KP_freeKernel_08_18 = ConstantKernelProblem(M;kernel=MLKernel.ConstantFreeKernel(M;m=2.0,g=0.8));


@time l_noKernel , sol_noKernel = trueLoss(KP_noKernel,tspan=100.,NTr=50,saveat=0.01);
@time l_freeKernel , sol_freeKernel = trueLoss(KP_freeKernel,tspan=100.,NTr=50,saveat=0.01);
@time l_freeKernel_08_18 , sol_freeKernel_08_18 = trueLoss(KP_freeKernel_08_18,tspan=100.,NTr=50,saveat=0.01);



### SAVING AND LO
const RESULT_DIR = "Results/AHO_FreeTheoryKernel/"

path_noKernel = joinpath(RESULT_DIR,"configs_noKernel.jld2")
path_freeKernel = joinpath(RESULT_DIR,"configs_freeKernel.jld2") 
path_freeKernel_08_18 = joinpath(RESULT_DIR,"configs_freeKernel_08_18.jld2") 


@save path_noKernel :sol=sol_noKernel :KP=KP_noKernel :l=l_noKernel
@save path_freeKernel :sol=sol_freeKernel :KP=KP_freeKernel :l=l_freeKernel
@save path_freeKernel_08_18 :sol=sol_freeKernel_08_18 :KP=KP_freeKernel_08_18 :l=l_freeKernel_08_18

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

begin
    lo_freeKernel_08_18 = load(path_freeKernel_08_18)
    sol_freeKernel_08_18 = lo_freeKernel_08_18[":sol"]
    KP_freeKernel_08_18 = lo_freeKernel_08_18[":KP"]
    l_freeKernel_08_18 = lo_freeKernel_08_18[":l"]
end


##### PLOTTING
fig_noKernel = MLKernel.plotSKContour(KP_noKernel,sol_noKernel)
fig_freeKernel = MLKernel.plotSKContour(KP_freeKernel,sol_freeKernel)
fig_freeKernel_08_18 = MLKernel.plotSKContour(KP_freeKernel_08_18,sol_freeKernel_08_18)

savefig(fig_noKernel,joinpath(RESULT_DIR,"RT_Observables_FreeTheory_noKernel.pdf"))
savefig(fig_freeKernel,joinpath(RESULT_DIR,"RT_Observables_FreeTheory_freeKernel.pdf"))
savefig(fig_freeKernel_08_18,joinpath(RESULT_DIR,"RT_Observables_FreeTheory_freeKernel_08_18.pdf"))


begin
fig = plot(;MLKernel.plot_setup(:topright)...)
fig = MLKernel.plotFWSKContour(KP_noKernel,sol_noKernel;fig=fig, plotSol=true,colors=[1,2])
fig = MLKernel.plotFWSKContour(KP_freeKernel,sol_freeKernel;fig=fig, plotSol=false,colors=[3,4])
fig = MLKernel.plotFWSKContour(KP_freeKernel_08_18,sol_freeKernel_08_18;fig=fig, plotSol=false,colors=[5,6])

fig.series_list[2+(1-1)*4+2][:label] = "NoKernel_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
fig.series_list[2+(1-1)*4+4][:label] = "NoKernel_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])

fig.series_list[2+(2-1)*4+2][:label] = "FreeKernel_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
fig.series_list[2+(2-1)*4+4][:label] = "FreeKernel_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])

fig.series_list[2+(3-1)*4+2][:label] = "FreeKernel_0.8_1.8_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
fig.series_list[2+(3-1)*4+4][:label] = "FreeKernel_0.8_1.8_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])
fig
end

savefig(fig,joinpath(RESULT_DIR,"CorrPlot_compare_FreeKernel.pdf"))



begin
    fig = plot(;MLKernel.plot_setup(:bottomright)...)
    fig = MLKernel.plot_x2_Contour(KP_noKernel,sol_noKernel;fig=fig, plotSol=true,colors=[1,2])
    fig = MLKernel.plot_x2_Contour(KP_freeKernel,sol_freeKernel;fig=fig, plotSol=false,colors=[3,4])
    fig = MLKernel.plot_x2_Contour(KP_freeKernel_08_18,sol_freeKernel_08_18;fig=fig, plotSol=false,colors=[5,6])
    
    fig.series_list[2+(1-1)*4+2][:label] = "NoKernel_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
    fig.series_list[2+(1-1)*4+4][:label] = "NoKernel_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])
    
    fig.series_list[2+(2-1)*4+2][:label] = "FreeKernel_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
    fig.series_list[2+(2-1)*4+4][:label] = "FreeKernel_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])
    
    fig.series_list[2+(3-1)*4+2][:label] = "FreeKernel_0.8_1.8_Re"#string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
    fig.series_list[2+(3-1)*4+4][:label] = "FreeKernel_0.8_1.8_Im"#string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])
    fig
end


savefig(fig,joinpath(RESULT_DIR,"x2Plot_compare_FreeKernel.pdf"))


