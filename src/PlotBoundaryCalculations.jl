export plotB


M_SIZE = 2.5
SOL_LW = 1.0

plot_setup() = plot_setup(:bottomright)

plot_setup(legend) = Dict(#:legend     =>  :bottomright,
                        :legend     =>  legend,
                        #:legend     => (1.25, 1.0), #:topleft,#(0.1, 0.8),#(1.08, 0.6),
                        :grid       => false,
                        :foreground_color_legend=> nothing,
                        :background_color_legend=> nothing,
                        :framestyle => :box,
                        :thickness_scaling => 1.2)

markers_dict(color,markershape) = Dict(
                        :color => color,
                        :markershape => markershape,
                        :markersize  => M_SIZE,
                        :markerstrokecolor => color,
                        :lw => false)

markers_dict(color) = markers_dict(color,:square) 

solution_line_dict = Dict(:color => "black",:lw => SOL_LW)





function plotB(B1::Array{Measurement{T}},BT::BoundaryTerms{M};plotType=:surface) where {T<:Real,M<:Model}#LM_AHO}
    
    @unpack Ys, Xs = BT

    if isnothing(Xs)
        fig = plot(xlabel=L"$Y$",ylabel=L"$B_1(Y)$",
                   #ylim=[-1.3,0.25]
                   ;plot_setup()...)
        plot!(fig,Ys,B1[1,:],label=L"$\textrm{Re}\langle L_c\;x^2 \rangle$";markers_dict(1)...)
        plot!(fig,Ys,B1[2,:],label=L"$\textrm{Im}\langle L_c\;x^2 \rangle$";markers_dict(2)...)
        plot!(fig,Ys,B1[3,:],label=L"$\textrm{Re}\langle L_c\;x \rangle$";markers_dict(3)...)
        plot!(fig,Ys,B1[4,:],label=L"$\textrm{Im}\langle L_c\;x \rangle$";markers_dict(4)...)
        return fig
    elseif plotType == :surface
        fig = plot(
        heatmap(Xs,Ys,Measurements.value.(B1[1,:,:]);
            title=L"$\textrm{Re}\langle L_c\;x(0)x(t) \rangle$",
            #title=L"$\textrm{Re}\langle L_c\;x^2 \rangle$",
            xlabel=L"$X$",ylabel=L"$Y$"),
        heatmap(Xs,Ys,Measurements.value.(B1[2,:,:]);
            title=L"$\textrm{Im}\langle L_c\;x(0)x(t) \rangle$",
            #title=L"$\textrm{Im}\langle L_c\;x^2 \rangle$",
            xlabel=L"$X$",ylabel=L"$Y$")
        )

        plot!(fig[1],palette=:vikO)
        plot!(fig[2],palette=:vikO)

        for i in 11:div(length(Xs),10):length(Xs)
            vline!(fig[1],[Xs[i]],color=i,legend=false,lw=2.0)
            vline!(fig[2],[Xs[i]],color=i,legend=false,lw=2.0)
        end
        return fig
    elseif plotType == :surfaceCutOut

        fig = plot(xlabel=L"$Y$",ylabel=L"$B_1(Y)$",palette=:vikO;plot_setup()...)
        fig2 = plot(xlabel=L"$Y$",ylabel=L"$B_1(Y)$",palette=:vikO,legendtitle=L"X";plot_setup((1.25, 1.0))...)

        hline!(fig,[0.],label=false,line=(0.5,:dash,1.0),color=:gray,)
        hline!(fig2,[0.],label=false,line=(0.5,:dash,1.0),color=:gray)


        for i in 11:div(length(Xs),10):length(Xs)
            plot!(fig,Ys,B1[1,:,i],label=false;markers_dict(i)...)
            plot!(fig2,Ys,B1[2,:,i],label=round(Int,Xs[i]);markers_dict(i)...)
        end
        
        return plot(fig,fig2,layout=grid(1, 2, widths=[0.43, 0.43]))
    elseif plotType == :surfaceAndLines
        figHeat = plotB(B1,BT)#;plotType=:surfaceCutOut)
        figLines = plotB(B1,BT;plotType=:surfaceCutOut)
        return plot(figHeat, figLines,size=(1000,600),layout= @layout [a ; b])
    end
end



function plotB(B1::Array{Measurement{T}},B2::Array{Measurement{T}},BT::BoundaryTerms{M}) where {T<:Real,M<:LM_AHO}
    
    @unpack Ys = BT

    fig = plot(xlabel=L"$Y$",ylabel=L"$B(Y)$",
                #ylim=[-0.05,0.05]
                ;plot_setup()...)
    plot!(fig,Ys,B1[1,:],label=L"$\textrm{Re}\langle L_c\;x^2 \rangle$";markers_dict(1)...)
    plot!(fig,Ys,B1[2,:],label=L"$\textrm{Im}\langle L_c\;x^2 \rangle$";markers_dict(2)...)
    plot!(fig,Ys,B2[1,:],label=L"$\textrm{Re}\langle L_c^2\;x^2 \rangle$";markers_dict(3)...)
    plot!(fig,Ys,B2[2,:],label=L"$\textrm{Im}\langle L_c^2\;x^2 \rangle$";markers_dict(4)...)
    return fig
    
end


function plotSKContour(KP::KernelProblem{AHO},sol)
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol))

    tp = KP.model.contour.tp[1:end-1]


    fig = plot(xlabel=L"$t_p$",#ylim=[-0.5,0.5]
                ;plot_setup(:bottomright)...)
    plot!(fig,tp,real(KP.y["x2"]),label=L"$\textrm{Solution}$";solution_line_dict...)
    plot!(fig,tp,imag(KP.y["x2"]),label=false;solution_line_dict...)
    
    scatter!(fig,tp,avgRe .± err_avgRe,label=L"$\textrm{Re}\langle x \rangle$";markers_dict(1)...)
    scatter!(fig,tp,avgIm .± err_avgIm,label=L"$\textrm{Im}\langle x \rangle$";markers_dict(2)...)
    
    scatter!(fig,tp,avg2Re .± err_avg2Re,label=L"$\textrm{Re}\langle x^2 \rangle$";markers_dict(3)...)
    scatter!(fig,tp,avg2Im .± err_avg2Im,label=L"$\textrm{Im}\langle x^2 \rangle$";markers_dict(4)...)

    plot!(fig,tp,real(KP.y["corr0t"]),label=false;solution_line_dict...)
    plot!(fig,tp,imag(KP.y["corr0t"]),label=false;solution_line_dict...)
    scatter!(fig,tp,corr0tRe .± err_corr0tRe,label=L"$\textrm{Re}\langle x(0)x(t) \rangle$";markers_dict(5)...)
    scatter!(fig,tp,corr0tIm .± err_corr0tIm,label=L"$\textrm{Im}\langle x(0)x(t) \rangle$";markers_dict(6)...)
    return fig
end

function plotFWSKContour(KP::KernelProblem{AHO},sol; plotSol=true, fig=nothing,colors=[1,2])
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol))
    
    #show(stdout,"text/plain",[corr0tRe err_corr0tRe corr0tIm err_corr0tIm])
    
    #maxinx = findall(diff(real(KP.model.contour.x0)) .< 0)[1]
    maxinx = argmax(real(KP.model.contour.x0))
    #maxinx = KP.model.contour.FWRTSteps
    x0 = real(KP.model.contour.x0[1:maxinx])

    if isnothing(fig)
        fig = plot(xlabel=L"$x_0$",#ylim=[-0.5,0.5]
                    ;plot_setup(:bottomright)...)
    end
    #plot!(fig,x0,real(KP.y["x2"]),label=L"$\textrm{Solution}$";solution_line_dict...)
    #plot!(fig,x0,imag(KP.y["x2"]),label=false;solution_line_dict...)
    
    #scatter!(fig,x0,avgRe[1:maxinx] .± err_avgRe[1:maxinx],label=L"$\textrm{Re}\langle x \rangle$";markers_dict(1)...)
    #scatter!(fig,x0,avgIm[1:maxinx] .± err_avgIm[1:maxinx],label=L"$\textrm{Im}\langle x \rangle$";markers_dict(2)...)
    
    #scatter!(fig,x0,avg2Re[1:maxinx] .± err_avg2Re[1:maxinx],label=L"$\textrm{Re}\langle x^2 \rangle$";markers_dict(3)...)
    #scatter!(fig,x0,avg2Im[1:maxinx] .± err_avg2Im[1:maxinx],label=L"$\textrm{Im}\langle x^2 \rangle$";markers_dict(4)...)

    if plotSol
        plot!(fig,x0,real(KP.y["corr0t"][1:maxinx]),label=false;solution_line_dict...)
        plot!(fig,x0,imag(KP.y["corr0t"][1:maxinx]),label=false;solution_line_dict...)
    end
    scatter!(fig,x0,corr0tRe[1:maxinx] .± err_corr0tRe[1:maxinx],label=L"$\textrm{Re}\langle x(0)x(t) \rangle$";markers_dict(colors[1])...)
    scatter!(fig,x0,corr0tIm[1:maxinx] .± err_corr0tIm[1:maxinx],label=L"$\textrm{Im}\langle x(0)x(t) \rangle$";markers_dict(colors[2])...)
    return fig
end



function plotFWSKContour(KP::KernelProblem{AHO},KP2::KernelProblem{AHO},sol)
    obs = calc_obs(KP,sol)
    avgRe, err_avgRe, avgIm, err_avgIm, avg2Re, err_avg2Re, avg2Im, err_avg2Im, corr0tRe, err_corr0tRe, corr0tIm, err_corr0tIm = calc_meanObs(KP,obs,length(sol))

    maxinx = findall(diff(real(KP.model.contour.x0)) .< 0)[1]
    x0 = real(KP.model.contour.x0[1:maxinx])

    maxinx2 = findall(diff(real(KP2.model.contour.x0)) .< 0)[1]
    x02 = real(KP2.model.contour.x0[1:maxinx2])


    fig = plot(xlabel=L"$x_0$",#ylim=[-0.5,0.5]
                ;plot_setup(:bottomright)...)
    #plot!(fig,x0,real(KP.y["x2"]),label=L"$\textrm{Solution}$";solution_line_dict...)
    #plot!(fig,x0,imag(KP.y["x2"]),label=false;solution_line_dict...)
    
    #scatter!(fig,x0,avgRe[1:maxinx] .± err_avgRe[1:maxinx],label=L"$\textrm{Re}\langle x \rangle$";markers_dict(1)...)
    #scatter!(fig,x0,avgIm[1:maxinx] .± err_avgIm[1:maxinx],label=L"$\textrm{Im}\langle x \rangle$";markers_dict(2)...)
    
    #scatter!(fig,x0,avg2Re[1:maxinx] .± err_avg2Re[1:maxinx],label=L"$\textrm{Re}\langle x^2 \rangle$";markers_dict(3)...)
    #scatter!(fig,x0,avg2Im[1:maxinx] .± err_avg2Im[1:maxinx],label=L"$\textrm{Im}\langle x^2 \rangle$";markers_dict(4)...)

    plot!(fig,x02,real(KP2.y["corr0t"][1:maxinx2]),label=false;solution_line_dict...)
    plot!(fig,x02,imag(KP2.y["corr0t"][1:maxinx2]),label=false;solution_line_dict...)
    scatter!(fig,x0,corr0tRe[1:maxinx] .± err_corr0tRe[1:maxinx],label=L"$\textrm{Re}\langle x(0)x(t) \rangle$";markers_dict(5)...)
    scatter!(fig,x0,corr0tIm[1:maxinx] .± err_corr0tIm[1:maxinx],label=L"$\textrm{Im}\langle x(0)x(t) \rangle$";markers_dict(6)...)
    return fig
end