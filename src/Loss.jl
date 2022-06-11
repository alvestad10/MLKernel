export trueLoss, calculateBVLoss


abstract type AbstractLoss end

struct LossVals_LM_AHO{Tobs,TID,TBT1,TBT2}
    avgRe::Vector{Tobs}
    avgIm::Vector{Tobs}
    avg2Re::Vector{Tobs}
    avg2Im::Vector{Tobs}
    avg3Re::Vector{Tobs}
    avg3Im::Vector{Tobs}
    imDrift::TID
    B1::TBT1
    B2::TBT2
end


struct Loss_LM_AHO{gLVType,gType,LType,LTrueType} <: AbstractLoss
    ID::Symbol
    Ys::Vector{Float64}
    get_LV::gLVType
    get_g::gType
    LTrain::LType
    LTrue::LTrueType
end

function trueLoss(KP::KernelProblem;tspan=30,NTr=10,saveat=0.01,sol=nothing)
    if isnothing(sol)
        sol = run_sim(KP,tspan=tspan,NTr=NTr;saveat=saveat)
    end
    return calcTrueLoss(sol,KP), sol
end


function calculateBVLoss(KP::KernelProblem;Ys=[10.0],tspan=50,NTr=30,sol=nothing)
    if isnothing(sol)
        sol = run_sim(KP,tspan=tspan,NTr=NTr)
    end

    BT = getBoundaryTerms(KP;Ys=Ys)
    B1,B2 = calcBoundaryTerms(sol,BT);

    #return sum(Ys' .* abs.(Measurements.value.(B1)).^2)#,sol
    #return sum( abs.(Measurements.value.(B1)).^2)#,sol
    if isnothing(B2)
        return sum( abs.(Measurements.value.(B1)).^2)
    else
        return sum( abs.(Measurements.value.(B1)).^2) + sum( abs.(Measurements.value.(B2)).^2) #,sol
    end
end
