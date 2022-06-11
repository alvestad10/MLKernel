using MLKernel
using JLD2
using Parameters
using Plots, LaTeXStrings

const RESULT_DIR = "Results/KernelForm"
if !isdir(RESULT_DIR)
    @warn string("RESULT_DIR: ", RESULT_DIR, " is not a directory. Constructing the path.")
    mkdir(RESULT_DIR)
end


@with_kw struct setupKernelForm
    ID::Integer
    kf::Symbol
    KDir::String 
    plotDir::String 
    lossID::Symbol                = :driftOpt
    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005),MLKernel.ADAM(0.0005)]
    iterations::Vector{Integer}   = [60,25,20]
    tspan                         = 30.
    NTr                           = 30
    saveat                        = 0.01
    M                             = AHO(1.,24.,1.5,1.0,10)
end 

function get_setupKernelForm(kf,dir)
    ID = MLKernel.get_highest_ID(dir) + 1
    KDir = joinpath(dir,string("KP_",ID))
    plotDir = joinpath(dir,string("Plots_",ID))
    

    sKF = setupKernelForm(ID=ID,kf=kf,KDir=KDir,plotDir=plotDir)
    sfunique = settingFileUnique(sKF,dir)
    return sKF, sfunique
end

function compareSettingFiles(sKF1::setupKernelForm,sKF2::setupKernelForm)
    !(sKF1.kf == sKF2.kf) && return false
    !(sKF1.lossID == sKF2.lossID) && return false

    if !(typeof(sKF1.opt) == typeof(sKF2.opt)) 
        return false
    else
        if (sKF1.opt isa Array) && (sKF2.opt isa Array)
            !(length(sKF1.opt) == length(sKF2.opt)) && return false
            !(all(typeof.(sKF1.opt) .== typeof.(sKF2.opt))) && return false
            !(all( [s.eta for s in sKF1.opt] .== [s.eta for s in sKF1.opt])) && return false
        elseif (sKF1.opt isa MLKernel.ADAM) && (sKF2.opt isa MLKernel.ADAM)
            !(sKF1.opt.eta == sKF2.opt.eta) && return false
        else
            return false
        end
    end
    !(sKF1.iterations == sKF2.iterations) && return false
    !(sKF1.tspan == sKF2.tspan) && return false
    !(sKF1.saveat == sKF2.saveat) && return false
    !MLKernel.isSameModel(sKF1.M,sKF2.M) && return false
    
    return true
end

function settingFileUnique(sKF::setupKernelForm,dir)

    files = readdir(dir;join=true)

    for file in files
        if string(split(split(file,"/")[end],"_")[1]) !== "Lookup"
            continue
        end

        lo = load(file)
        file_sKF = lo["sKF"]
        if compareSettingFiles(sKF,file_sKF) && isfile(string(split(file,"Lookup")[1],"Result",split(file,"Lookup")[end]))
            return parse(Int,split(split(file,".")[1],"_")[end])
        end
    end
    return 0
end




function saveSetting(sKF::setupKernelForm,dir)
    save(joinpath(dir,string("Lookup_",sKF.ID,".jld2")),"sKF",sKF)
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





function run(kernelTypes)
    for kt in kernelTypes
        sKF, loadOldID = get_setupKernelForm(kt,RESULT_DIR)

        if loadOldID != 0
            println("Found old version of ", kt)
            continue
        else
            !isdir(sKF.KDir) && mkdir(sKF.KDir)
            !isdir(sKF.plotDir) && mkdir(sKF.plotDir)
        end



        res = Dict(:LdriftOpt => Float64[], 
            :LTrue => Float64[], 
            :LSym => Float64[])
        
        saveSetting(sKF,RESULT_DIR)
        KP = ConstantKernelProblem(sKF.M;kernel=MLKernel.ConstantKernel(sKF.M,kernelType=kt));

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

            saveK = MLKernel.SaveKernel(sKF.KDir,sKF.plotDir)
            MLKernel.savefig(fig,joinpath(sKF.plotDir,string("observables","_",saveK.KIDMax,".pdf")))
            MLKernel.saveKP(KP,saveK)
        end
        
        for (i,iter) in enumerate(sKF.iterations)
            tspan = (isempty(size(sKF.tspan)) ? sKF.tspan : sKF.tspan[i])
            NTr = (isempty(size(sKF.NTr)) ? sKF.NTr : sKF.NTr[i])
            opt = (isempty(size(sKF.opt)) ? sKF.opt : sKF.opt[i])
            saveat = (isempty(size(sKF.saveat)) ? sKF.saveat : sKF.saveat[i])
            LK = MLKernel.LearnKernel(KP,sKF.lossID,iter; tspan=tspan,NTr=NTr,
                                saveat=saveat,
                                opt=opt,
                                runs_pr_epoch=1);
            
            learnKernel(LK; cb=cb)
        end

        saveResult(res,RESULT_DIR,sKF.ID)

    end 
end

function plotResults(kernelTypes)
    
    fig = plot(;xlabel="Iteration",yaxis=:log,MLKernel.plot_setup(:top)...)#, ylim=[1e-2,Inf])

    for (i,kt) in enumerate(kernelTypes)
        sKF, loadOldID = get_setupKernelForm(kt,RESULT_DIR)

        if loadOldID != 0
            res = loadResult(RESULT_DIR,loadOldID)
            
            if res == 0
                @warn "Did not find a result for $kt"
                continue
            end

            name = kernelTypeName(sKF.kf)
            #plot!(fig,res[:LTrue] .- res[:LSym], 
            #        label=string(L"$L_{\textrm{True}} - L_{\textrm{Sym}}$", ", K=$name"),
            #        color=i,lw=1.5)
            plot!(fig,res[:LTrue], 
                    label=string(L"$L_{\textrm{True}}$", ", K=$name"),
                    color=i,lw=1.5)
            #plot!(fig,res[:LdriftOpt])
            plot!(fig,res[:LSym], 
                label=string(L"$L_{\textrm{Sym}}$", ", K=$name"),
                linestyle=:dot,color=i,lw=1.5)
        end
    end
    savefig(fig,joinpath(RESULT_DIR,"LPlot_602520.pdf"))
    display(fig)
end

function plotBestLSymKernel(kernelTypes)
    bestLSym = 1.
    bestInx = 0
    bestkt = :K

    for (i,kt) in enumerate(kernelTypes)
        _, loadOldID = get_setupKernelForm(kt,RESULT_DIR)
        if loadOldID != 0
            res = loadResult(RESULT_DIR,loadOldID)
            
            if res == 0
                @warn "Did not find a result for $kt"
                continue
            end
            
            if minimum(res[:LSym]) < bestLSym
                bestLSym = minimum(res[:LSym])
                bestInx = argmin(res[:LSym])
                bestkt = kt
            end
            
        end
    end

    @show bestInx, bestkt

    _, loadOldID = get_setupKernelForm(bestkt,RESULT_DIR)
    KP = MLKernel.loadKP(bestInx,loadOldID,RESULT_DIR)

    sol = run_sim(KP,tspan=100,NTr=100)
    LdriftOpt = MLKernel.calcDriftOpt(sol,KP)
    TLoss = MLKernel.calcTrueLoss(sol,KP)
    LSym =  MLKernel.calcSymLoss(sol,KP)
    println("LDriftOpt: ", round(LdriftOpt,digits=5), ", TLoss: ", round(TLoss,digits=5), ", LSym: ", round(LSym,digits=5))
    display(MLKernel.plotSKContour(KP,sol))
end

kernelTypes = [:K]#,:expiP,:expA]#,:expiHerm,:expiSym]
kernelTypeName(kernelType::Symbol) = begin
    if kernelType == :K
        return "K"
    elseif kernelType == :expiP
        return "expiP"
    elseif kernelType == :expA
        return "expA"
    elseif kernelType == :expiHerm
        return "expiHerm"
    elseif kernelType == :expiSym
        return "expiSym"
    end
end
run(kernelTypes)
plotResults(kernelTypes)
plotBestLSymKernel(kernelTypes)


# Previous runs:
# 
# kernelTypes = [:K,:expiP,:expA]
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
#    iterations::Vector{Integer}   = [50,50]
#    tspan                         = 30.
#    NTr                           = 30
#    saveat                        = 0.01
#    M                             = AHO(1.,24.,1.5,1.0,10)
#
# kernelTypes = [:K,:expiP,:expA]
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005)]
#    iterations::Vector{Integer}   = [30,30]
#    tspan                         = 30.
#    NTr                           = 30
#    saveat                        = 0.01
#    M                             = AHO(1.,24.,1.5,1.0,10)
#
#
# kernelTypes = [:K,:expiP,:expA,:expiHerm,:expiSym]
#    lossID::Symbol                = :driftOpt
#    opt                           = [MLKernel.ADAM(0.01),MLKernel.ADAM(0.005),MLKernel.ADAM(0.0005)]
#    iterations::Vector{Integer}   = [60,25,20]
#    tspan                         = 30.
#    NTr                           = 30
#    saveat                        = 0.01
#    M                             = AHO(1.,24.,1.5,1.0,10)



