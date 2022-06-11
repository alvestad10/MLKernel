using MLKernel
using JLD2
using Parameters
using Plots, LaTeXStrings

const RUN = "AlexandruSK_2550_expA_10_15_20_30"
const RESULT_DIR = joinpath("Results/DifferentRealTimes",RUN)
const RESULTPLOT_DIR = joinpath(RESULT_DIR,"ResultPlots")
if !isdir(RESULT_DIR)
    @warn string("RESULT_DIR: ", RESULT_DIR, " is not a directory. Constructing the path.")
    mkdir(RESULT_DIR)
    mkdir(RESULTPLOT_DIR)
end


@with_kw struct setupDRT
    ID::Integer
    M::AHO              
    KDir::String 
    plotDir::String 
    kf::Symbol                    = :expA
    lossID::Symbol                = :driftOpt
    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
    iterations::Vector{Integer}   = [25,50]
    tspan                         = 30.
    NTr                           = 20
    saveat                        = 0.01
end 

function get_setupDRT(rt,dir)
    ID = MLKernel.get_highest_ID(dir) + 1
    KDir = joinpath(dir,string("KP_",ID))
    plotDir = joinpath(dir,string("Plots_",ID))
    
    M = AHO(1.,24.,rt,1.0,10;Δβ=0.5)
    S = setupDRT(ID=ID,M=M,KDir=KDir,plotDir=plotDir)
    sfunique = settingFileUnique(S,dir)
    return S, sfunique
end

function compareSettingFiles(S1::setupDRT,S2::setupDRT)
    
    !MLKernel.isSameModel(S1.M,S2.M) && return false
    !(S1.kf == S2.kf) && return false
    !(S1.lossID == S2.lossID) && return false

    if !(typeof(S1.opt) == typeof(S2.opt)) 
        return false
    else
        if (S1.opt isa Array) && (S2.opt isa Array)
            !(length(S1.opt) == length(S2.opt)) && return false
            !(all(typeof.(S1.opt) .== typeof.(S2.opt))) && return false
            !(all( [s.eta for s in S1.opt] .== [s.eta for s in S1.opt])) && return false
        elseif (S1.opt isa MLKernel.ADAM) && (S2.opt isa MLKernel.ADAM)
            !(S1.opt.eta == S2.opt.eta) && return false
        else
            return false
        end
    end
    !(S1.iterations == S2.iterations) && return false
    !(S1.tspan == S2.tspan) && return false
    !(S1.saveat == S2.saveat) && return false
    
    return true
end

function settingFileUnique(S::setupDRT,dir)

    files = readdir(dir;join=true)

    for file in files
        if string(split(split(file,"/")[end],"_")[1]) !== "Lookup"
            continue
        end

        lo = load(file)
        file_S = lo["S"]
        if compareSettingFiles(S,file_S) && isfile(string(split(file,"Lookup")[1],"Result",split(file,"Lookup")[end]))
            return parse(Int,split(split(file,".")[1],"_")[end])
        end
    end
    return 0
end




function saveSetting(S::setupDRT,dir)
    save(joinpath(dir,string("Lookup_",S.ID,".jld2")),"S",S)
end

function saveResult(res,dir,ID)
    save(joinpath(dir,string("Result_",ID,".jld2")),"res",res)
end

function loadResult(dir,ID)
    try
        return load(joinpath(dir,string("Result_",ID,".jld2")))["res"]
    catch ArgumentError
        @warn string(ID, " does not have a result file")
        return 0
    end

    return 
end





function run(DRTs)
    for rt in DRTs
        S, loadOldID = get_setupDRT(rt,RESULT_DIR)

        if loadOldID != 0
            println("Found old version of ", rt)
            continue
        else
            !isdir(S.KDir) && mkdir(S.KDir)
            !isdir(S.plotDir) && mkdir(S.plotDir)
        end



        res = Dict(:LdriftOpt => Float64[], 
            :LTrue => Float64[], 
            :LSym => Float64[])
        
        saveSetting(S,RESULT_DIR)
        KP = ConstantKernelProblem(S.M;kernel=MLKernel.ConstantKernel(S.M,kernelType=S.kf));

        cb(LK::LearnKernel;sol=nothing,addtohistory=false) = begin
            KP = LK.KP
            if isnothing(sol)
                sol = run_sim(KP,tspan=LK.tspan_test,NTr=LK.NTr_test)
            end
            
            LdriftOpt = MLKernel.calcDriftOpt(sol,KP)
            TLoss = LK.Loss.LTrue(sol,KP)
            LSym =  MLKernel.calcSymLoss(sol,KP)
            println("LDriftOpt: ", round(LdriftOpt,digits=5), ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5))

            addtohistory && begin append!(res[:LdriftOpt],LdriftOpt); append!(res[:LTrue],TLoss); append!(res[:LSym],LSym)
                            end
            
            fig = MLKernel.plotSKContour(KP,sol)
            #display(fig)

            saveK = MLKernel.SaveKernel(S.KDir,S.plotDir)
            MLKernel.savefig(fig,joinpath(S.plotDir,string("observables","_",saveK.KIDMax,".pdf")))
            MLKernel.saveKP(KP,saveK)
        end
        
        for (i,iter) in enumerate(S.iterations)
            tspan = (isempty(size(S.tspan)) ? S.tspan : S.tspan[i])
            NTr = (isempty(size(S.NTr)) ? S.NTr : S.NTr[i])
            opt = (isempty(size(S.opt)) ? S.opt : S.opt[i])
            saveat = (isempty(size(S.saveat)) ? S.saveat : S.saveat[i])
            LK = MLKernel.LearnKernel(KP,S.lossID,iter; tspan=tspan,NTr=NTr,
                                saveat=saveat,
                                opt=opt,
                                runs_pr_epoch=1);
            
            learnKernel(LK; cb=cb)
        end

        saveResult(res,RESULT_DIR,S.ID)

    end 
end

function plotResults(DRTs)
    
    fig = plot(;xlabel="Iteration",yaxis=:log,MLKernel.plot_setup(:top)...)#, ylim=[1e-2,Inf])

    for (i,rt) in enumerate(DRTs)
        _, loadOldID = get_setupDRT(rt,RESULT_DIR)

        if loadOldID != 0
            res = loadResult(RESULT_DIR,loadOldID)
            
            if res == 0
                @warn "Did not find a result for $rt"
                continue
            end

            #plot!(fig,res[:LTrue] .- res[:LSym], 
            #        label=string(L"$L_{\textrm{True}} - L_{\textrm{Sym}}$", ", K=$name"),
            #        color=i,lw=1.5)
            plot!(fig,res[:LTrue], 
                    label=L"L_{\textrm{True}}, \; rt=%$rt",
                    color=i,lw=1.5)
            #plot!(fig,res[:LdriftOpt])
            plot!(fig,res[:LSym], 
                label=L"L_{\textrm{Sym}}, \; rt=%$rt",
                linestyle=:dot,color=i,lw=1.5)
        end
    end
    savefig(fig,joinpath(RESULTPLOT_DIR,string("LPlot",RUN,".pdf")))
    display(fig)
end

function plotBestLSymKernel(DRTs)
    bestLSym = ones(length(DRTs))
    bestInx = zeros(Integer,length(DRTs))

    fig = plot(;MLKernel.plot_setup(:topright)...)

    for (i,rt) in enumerate(sort(DRTs,rev=true))
        _, loadOldID = get_setupDRT(rt,RESULT_DIR)
        if loadOldID != 0
            res = loadResult(RESULT_DIR,loadOldID)
            
            if res == 0
                @warn "Did not find a result for $rt"
                continue
            end
            
            if minimum(res[:LSym]) < bestLSym[i]
                bestLSym[i] = minimum(res[:LSym])
                bestInx[i] = argmin(res[:LSym])
            end
            
        else
            continue
        end


        if rt==2.0 
            bestInx[i] = 62
        end
        KP = MLKernel.loadKP(bestInx[i],loadOldID,RESULT_DIR)

        sol = run_sim(KP,tspan=30,NTr=100,saveat=0.01)
        LdriftOpt = MLKernel.calcDriftOpt(sol,KP)
        TLoss = MLKernel.calcTrueLoss(sol,KP)
        LSym =  MLKernel.calcSymLoss(sol,KP)
        println(rt,", LDriftOpt: ", round(LdriftOpt,digits=5), ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5))
        
        fig = MLKernel.plotFWSKContour(KP,sol;fig=fig, plotSol= (rt == maximum(DRTs)),colors=(i-1)*2 .+ [1,2])

        fig2 = MLKernel.plotSKContour(KP,sol)
        savefig(fig,joinpath(RESULTPLOT_DIR,string("ObservablePlotBestLSym_",rt,"_",RUN,".pdf")))
        display(fig2)

    end
    for (i,rt) in enumerate(sort(DRTs,rev=true))
        fig.series_list[2+(i-1)*4+2][:label] = string(rt, ": ",fig.series_list[2+(i-1)*4+2][:label])
        #fig.series_list[2+(i-1)*4+2][:markercolor] = theme_palette(:auto)[(i-1)*2+1]
        #fig.series_list[2+(i-1)*4+2][:markerstrokecolor] = theme_palette(:auto)[(i-1)*2+1]
        fig.series_list[2+(i-1)*4+4][:label] = string(rt, ": ",fig.series_list[2+(i-1)*4+4][:label])
        #fig.series_list[2+(i-1)*4+4][:markercolor] = theme_palette(:auto)[(i-1)*2+2]
        #fig.series_list[2+(i-1)*4+4][:markerstrokecolor] = theme_palette(:auto)[(i-1)*2+2]
    end
    savefig(fig,joinpath(RESULTPLOT_DIR,string("BestCorrPlot_",RUN,".pdf")))
    display(fig)


    println.(sort(DRTs,rev=true),": i=",bestInx,", BestLSym=",bestLSym)
end

realTimes = [1.0,1.5,2.0]#,3.0,4.0]
run(realTimes)
plotResults(realTimes)
plotBestLSymKernel(realTimes)

s
# Previous runs:
# 
# realTimes = [1.0,1.5,2.0]
# M = AHO(1.,24.,rt,1.0,8)
#    kf::Symbol                    = :expiP
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01)]
#    iterations::Vector{Integer}   = [60]
#    tspan                         = 10.
#    NTr                           = 10
#    saveat                        = 0.01
#
# realTimes = [1.0,1.5,2.0]
# M = AHO(1.,24.,rt,1.0,8)
#    kf::Symbol                    = :expA
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
#    iterations::Vector{Integer}   = [50]
#    tspan                         = 20.
#    NTr                           = 20
#    saveat                        = 0.01
# 
# Alexandru contour
# realTimes = [1.0,1.5,2.0,3.0,4.0]
# M = AHO(1.,24.,rt,1.0,10; Δβ=0.5)
#    kf::Symbol                    = :expA
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
#    iterations::Vector{Integer}   = [25,50]
#    tspan                         = 30.
#    NTr                           = 20
#    saveat                        = 0.01
#
# Alexandru contour with more points
# realTimes = [1.0,1.5,2.0,3.0,4.0]
# M = AHO(1.,24.,rt,1.0,14; Δβ=0.5)
#    kf::Symbol                    = :expA
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
#    iterations::Vector{Integer}   = [25,50]
#    tspan                         = 30.
#    NTr                           = 20
#    saveat                        = 0.01



