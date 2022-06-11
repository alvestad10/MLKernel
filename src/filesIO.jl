using JLD2

function get_highest_ID(dir)

    getID(f) = begin
        try 
            return parse(Int64,split(split(f,".")[1],"_")[2])
        catch ArgumentError
            return 0
        end
    end
    return length(readdir(dir)) > 0 ? maximum(getID,readdir(dir)) : 0
end


struct SaveKernel
    resultKernelDir::String
    resultPlotDir::String
    KIDMax::Integer

    function SaveKernel(resultKernelDir,resultPlotDir)
        maxID = get_highest_ID(resultKernelDir)
        new(resultKernelDir,resultPlotDir,maxID+1)
    end
end


function saveKP(KP::KernelProblem,sf::SaveKernel)
    path = joinpath(sf.resultKernelDir,string("KP_",sf.KIDMax,".jld2"))
    save(path,"KP",KP)
end

function loadKP(ID,sf::SaveKernel)
    return loadKP(ID,sf.resultKernelDir)
end

function loadKP(ID,runID,dir)
    kdir = joinpath(dir,string("KP_",runID))
    return loadKP(ID,kdir)
end


function loadKP(ID,kernelDir)
    path = joinpath(kernelDir,string("KP_",ID,".jld2"))
    return load(path)["KP"]
end

