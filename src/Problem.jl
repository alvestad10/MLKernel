export KernelProblem, ConstantKernelProblem, FieldDependentKernelProblem, NNDependentKernelProblem,NNDependentKernel1Problem
export run_sim

struct KernelProblem{MType<:Model, KType<:Kernel,AType,BType,yType}
    model::MType
    kernel::KType
    a::AType
    b::BType
    y::yType
end

function ConstantKernelProblem(model::Model;kernel=ConstantKernel(model))

    a_func!, b_func! = get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end

function FieldDependentKernelProblem(model::Model;kwargs...)
    
    kernel = FieldDependentKernel(model;kwargs...)
    
    a_func!, b_func! = get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end

function NNDependentKernelProblem(model::Model; kwargs...)
    
    kernel = NNDependentKernel(model;kwargs...)
    
    a_func!, b_func! = get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end

function NNDependentKernel1Problem(model::Model; kwargs...)
    
    kernel = NNDependentKernel1(model;kwargs...)
    
    a_func!, b_func! = get_ab(model,kernel)

    y = getSolutions(model)

    return KernelProblem(model,kernel,a_func!,b_func!,y)
end


function updateProblem(KP::KernelProblem)
    @unpack model, kernel, a, b, y = KP

    updateKernel!(kernel.pK)
    a_func!, b_func! = get_ab(model,kernel)

    return KernelProblem(model, kernel, a_func!, b_func!, y) 
end





function run_sim(KP::KernelProblem; tspan=20, NTr = 10, saveat=0.01)

    @unpack kernel, a, b, model = KP
    K = getKernelParamsSim(kernel)

    if model isa AHO 
        u0 = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps)
        noise_rate_prototype = zeros(eltype(getKernelParams(kernel)),2*model.contour.t_steps,model.contour.t_steps)
    else
        u0 = zeros(eltype(K),2)
        noise_rate_prototype = zeros(2,1)
    end

    function prob_init_func(prob,i,repeat)
        if model isa AHO
            prob.u0 .= [rand(model.contour.t_steps); zeros(model.contour.t_steps)]
        else
            prob.u0 .= [randn();0.]
        end
        prob
    end

    tspan_init=1
    prob_init = SDEProblem(a,b,u0,(0.0,tspan_init),K, noise_rate_prototype=noise_rate_prototype)
    ensembleProb_init = EnsembleProblem(prob_init,prob_func=prob_init_func)
    sol_init = solve(#prob_init,
                ensembleProb_init,
                #LambaEM(),
                #SOSRA(),
                ImplicitEM(theta=1.0,autodiff=true),
                #ISSEM(theta=1.0,autodiff=false),
                EnsembleThreads(); trajectories=NTr, 
                dt=1e-3, saveat=[tspan_init], #save_start=false,
                dtmax=1e-3,abstol=1e-2,reltol=1e-2)

    function prob_func(prob,i,repeat)
        prob.u0 .= sol_init[i].u[end]
        prob
    end

    prob = SDEProblem(a,b,u0,(0.0,tspan),K,
                        noise_rate_prototype=noise_rate_prototype)
    ensembleProb = EnsembleProblem(prob,prob_func=prob_func)
    return solve(ensembleProb,
                #LambaEM(),
                #SKenCarp(ode_error_est=false),
                #SOSRA(),
                #DRI1(),
                ImplicitEM(theta=0.6,autodiff=true),
                #ISSEM(theta=1.0,autodiff=true),
                EnsembleThreads(); trajectories=NTr, 
                maxiters=1e10,
                #adaptive=false,
                dt=5e-4, saveat=saveat, save_start=false,
                #save_noise=true,
                dtmax=1e-3,abstol=1e-2,reltol=1e-2)
end



