using Plots
using LaTeXStrings
using Printf

const dataPath = joinpath("..","..","data")
const imagePath = joinpath(".","Images")

const types = [:kf, :kfaem, :raukf]
const typePaths = Dict(id => String(id) for id in types)
typeNames = Dict(id => String(id) for id in types)
typeNames[:kf] = "KF"
typeNames[:kfaem] = "KF-AEM"
typeNames[:raukf] = "UKF"
const typeParms = [:Lx, :Ly, :Lz, :Lt, :Lxy, :Lxyz, :Lfile]
const typeValues = [:t, :T, :cT, :Q, :cQ, :Tm, :cTm, :TsN, :Ts, :Qs]

# [Lx, Ly, Lz, Lt, Lxy, Lxyz, Lfile]
dataParms = Dict(id => Array{Int,1}(undef,size(typeParms,1)) for id in types)
dataSizes = Dict(id => Dict(idd => Int(0) for idd in typeValues) for id in types)
dataOffset = Dict(id => Dict(idd => Int(0) for idd in typeValues) for id in types)

#[t T cT Q cQ Tm cTm Ts Qs]
dataValues = Dict(id => Dict(idd => Array{Float64,1}() for idd in typeValues) for id in types)

for id in types
    aux = Array{Int32}(undef,4)
    read!(joinpath(dataPath,typePaths[id],string("Parms.bin")),aux)
    dataParms[id][1:4] = aux[1:4]
    dataParms[id][5] = aux[1] * aux[2] # Lxy
    dataParms[id][6] = aux[1] * aux[2] * aux[3] # Lxyz
    dataParms[id][7] =  2 * aux[1] * aux[2] * aux[3] + 7 * aux[1] * aux[2] + 1 # Lfile
    offset = 0
    for (i,s) in enumerate(typeValues) 
        temp = 0
        if s == :t
            temp = 1
        elseif s == :T || s == :cT
            temp = dataParms[id][6]
        elseif s == :Q || s == :cQ
            temp = dataParms[id][5]
        elseif s == :Tm || s == :cTm
            temp = dataParms[id][5]
        elseif s == :TsN || s == :Ts || s == :Qs
            temp = dataParms[id][5]
        else 
            println("Symbol ",s," Not defined")
        end
        dataOffset[id][s] = offset
        dataSizes[id][s] = temp
        offset = offset + temp
    end
end

function getData(id,parms,values,valuesSizes,valuesOffsets,typeValues,mainPath,typePaths)
    path = joinpath(mainPath, typePaths[id])
    names = readdir(joinpath(path,"ready"), join=false)
    if length(names) > 0
        iStr = sortperm(parse.(Int32,names))
        println(path," ",parms[id][7])
        for index in iStr
            name = names[index]
            data = Array{Float64}(undef,parms[id][7])
            read!(joinpath(path,string("Values", name, ".bin")), data)
            for (i,s) in enumerate(typeValues)
                offset = valuesOffsets[id][s]
                len = valuesSizes[id][s]
                append!(values[id][s],data[offset+1:offset+len])
            end
        end
        return true
    end
    return false
end

function cleanLists(types,typeValues,dataValues)
    for id in types
        for s in typeValues
            empty!(dataValues[id][s])
        end
    end
end

function getTicks(max,min,sdiv,format=:default)
    dc = (max - min) / sdiv
    if dc == 0.0
        dc = 1
        r = min-dc:dc:max+dc
    else 
        r = min:dc:max
    end
    if format == :scientific
        f = Ref(Printf.Format("%.2E"))
        s = Printf.format.(f,r)
        s = replace.(s,"E" => "\u00D710^{")
        s = replace.(s,"+" => "")
        s = s .* "}"
        s = replace.(s,"10^{0" => "10^{")
        s = replace.(s,"10^{}" => "")
        s = replace.(s,"-" => "\u2212")
    else
        s = string.(r)
    end
    return (r,s)
end


function printGraphs()
    cleanLists(types,typeValues,dataValues)
    for id in types
        if getData(id,dataParms,dataValues,dataSizes,dataOffset,typeValues,dataPath,typePaths)
            for t in [49 249 499]
                # Profile Graphs
                Lx = dataParms[id][1]
                Ly = dataParms[id][2]
                offsetXY = t * Lx * Ly + 1
                Sx = 0.12
                Sy = 0.12
                dx2 = Sx/Lx/2
                dy2 = Sy/Ly/2
                amp = 5*10^4
                x = range(0+dx2,Sx-dx2,Lx)
                y = range(0+dy2,Sy-dy2,Ly)
                zT = reshape(dataValues[id][:Tm][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly)
                zQ = reshape(dataValues[id][:Q][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly) * amp
                zcT = sqrt.(reshape(dataValues[id][:cTm][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly))
                zcQ = sqrt.(reshape(dataValues[id][:cQ][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly)) * amp
                zTsN = reshape(dataValues[id][:TsN][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly)
                zTs = reshape(dataValues[id][:Ts][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly)
                zQs = reshape(dataValues[id][:Qs][offsetXY:offsetXY+Lx*Ly-1],Lx,Ly) * amp

                colgrad = cgrad(:thermal, rev = false)

                plot_font = "Computer Modern"
                default(fontfamily=plot_font)
                graphTProfile = heatmap(x,y,zT,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zT),minimum(zT),8,:scientific),title="Temperature Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title ="Temperature [K]",dpi=1000)
                savefig(graphTProfile,joinpath(imagePath,"TemperatureProfile"*typePaths[id]*string(t)*".pdf"))
                graphcTProfile = heatmap(x,y,zcT,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zcT),minimum(zcT),8,:scientific),title="Temperature's Standard Deviation Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Temperature [K]",dpi=1000)
                savefig(graphcTProfile,joinpath(imagePath,"TemperatureCovarianceProfile"*typePaths[id]*string(t)*".pdf"))
                graphQProfile = heatmap(x,y,zQ,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zQ),minimum(zQ),8,:scientific),title="Heat Flux Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Heat Flux [W/m" * L"^2" *"]",dpi=1000)
                savefig(graphQProfile,joinpath(imagePath,"HeatFluxProfile"*typePaths[id]*string(t)*".pdf"))
                graphcQProfile = heatmap(x,y,zcQ,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zcQ),minimum(zcQ),8,:scientific),title="Heat Flux's Standard Deviation Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Heat Flux [W/m" * L"^2" *"]",dpi=1000)
                savefig(graphcQProfile,joinpath(imagePath,"HeatFluxCovarianceProfile"*typePaths[id]*string(t)*".pdf"))
                graphTsNProfile = heatmap(x,y,zTsN,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zTsN),minimum(zTsN),8,:scientific),title="Synthetic Temperature Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Temperature [K]",dpi=1000)
                savefig(graphTsNProfile,joinpath(imagePath,"TemperatureSyntheticNoiseProfile"*typePaths[id]*string(t)*".pdf"))
                graphTsProfile = heatmap(x,y,zTs,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zTs),minimum(zTs),8,:scientific),title="Synthetic Temperature Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Temperature [K]",dpi=1000)
                savefig(graphTsProfile,joinpath(imagePath,"TemperatureSyntheticProfile"*typePaths[id]*string(t)*".pdf"))
                graphQsProfile = heatmap(x,y,zQs,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zQs),minimum(zQs),8,:scientific),title="Synthetic Heat Flux Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Heat Flux [W/m" * L"^2" *"]",dpi=1000)
                savefig(graphQsProfile,joinpath(imagePath,"HeatFluxSyntheticProfile"*typePaths[id]*string(t)*".pdf"))
                graphTResidueProfile = heatmap(x,y,zTs-zT,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zTs-zT),minimum(zTs-zT),8,:scientific),title="Temperature's Residue Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Temperature [K]",dpi=1000)
                savefig(graphTResidueProfile,joinpath(imagePath,"TemperatureResidueProfile"*typePaths[id]*string(t)*".pdf"))
                graphQResidueProfile = heatmap(x,y,zQs-zQ,xlims=(0,Sx),ylims=(0,Sy),yflip=false,c=colgrad,aspect_ratio=:equal,xticks=getTicks(Sy,0,6),yticks=getTicks(Sy,0,6),colorbar_ticks=getTicks(maximum(zQs-zQ),minimum(zQs-zQ),8,:scientific),title="Heat Flux's Residue Profile",xlabel="X [m]",ylabel="Y [m]",colorbar_title="Heat Flux [W/m" * L"^2" *"]",dpi=1000)
                savefig(graphQResidueProfile,joinpath(imagePath,"HeatFluxResidueProfile"*typePaths[id]*string(t)*".pdf"))
            end
        end
    end
    cleanLists(types,typeValues,dataValues)
    graphTEvolution = Plots.plot(dpi=1000)
    graphTZoomEvolution = Plots.plot(dpi=1000)
    graphQEvolution = Plots.plot(dpi=1000)
    graphTsEvolution = Plots.plot(dpi=1000)
    graphQsEvolution = Plots.plot(dpi=1000)
    graphTResidueEvolution = Plots.plot(dpi=1000)
    graphQResidueEvolution = Plots.plot(dpi=1000)
    plotref = true
    color = palette(:tab10,length(types))
    for (k,id) in enumerate(types)
        if getData(id,dataParms,dataValues,dataSizes,dataOffset,typeValues,dataPath,typePaths)
            
            # Temporal Graphs
            Lx = dataParms[id][1]
            Ly = dataParms[id][2]
            St = 10.0
            amp = 5*10^4
            ix = Int(Lx/2)
            iy = Int(Ly/2)
            i = iy * Lx + ix
            t = dataValues[id][:t][begin:1:end]
            zT = dataValues[id][:Tm][begin+i:Lx*Ly:end]
            zQ = dataValues[id][:Q][begin+i:Lx*Ly:end] * amp
            zcT = sqrt.(dataValues[id][:cTm][begin+i:Lx*Ly:end])
            zcQ = sqrt.(dataValues[id][:cQ][begin+i:Lx*Ly:end]) * amp
            zTs = dataValues[id][:Ts][begin+i:Lx*Ly:end]
            zTsN = dataValues[id][:TsN][begin+i:Lx*Ly:end]
            zQs = dataValues[id][:Qs][begin+i:Lx*Ly:end] * amp
            
            nT = 6
            Tinf = 250
            Tsup = 1900
            TZoominf = 270
            TZoomsup = 360
            dT = (Tsup - Tinf)/nT
            dTZoom = (TZoomsup - TZoominf)/nT
            rTinf = -20
            rTsup = 20
            
            nQ = 8
            Qinf = -20*amp
            Qsup = 140*amp
            dQ = (Qsup - Qinf)/nQ
            rQinf = -Qsup
            rQsup = Qsup

            if plotref
                plot!(graphTEvolution,t,zTs,xlims=(0,St),yflip=false,title="Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label="Reference",color=:black)
                plot!(graphTZoomEvolution,t,zTs,xlims=(0.07*St,0.14*St),yflip=false,title="Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label="Reference",color=:black)
                plot!(graphTZoomEvolution,t,zTsN,xlims=(0.07*St,0.14*St),yflip=false,title="Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label="Noisy Reference",color=:black,linestyle=:dashdot,linewidth=0.5)
                plot!(graphQEvolution,t,zQs,xlims=(0,St),yflip=false,title="Heat Flux Profile",xlabel="Time [s]",ylabel="Heat Flux [W/m" * L"^2" *"]",label="Reference",color=:black)
                plot!(graphTsEvolution,t,zTs,xlims=(0,St),ylims=(Tinf,Tsup),yticks=Tinf:dT:Tsup,yflip=false,title="Synthetic Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label="Reference",color=color[k],linestyle=:solid,marker=:xcross,markersize=2)
                plot!(graphQsEvolution,t,zQs,xlims=(0,St),ylims=(Qinf,Qsup),yticks=Qinf:dQ:Qsup,yflip=false,title="Synthetic Heat Flux Profile",xlabel="Time [s]",ylabel="Heat Flux [W/m" * L"^2" *"]",label="Reference",color=color[k],linestyle=:solid,marker=:xcross,markersize=2)
                plotref = false
            end
            plot_font = "Computer Modern"
            default(fontfamily=plot_font)
            plot!(graphTEvolution,t,[zT zT-1.96.*zcT zT+1.96.*zcT],xlims=(0,St),ylims=(Tinf,Tsup),yticks=Tinf:dT:Tsup,yflip=false,title="Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label=[typeNames[id] "" ""],color=[color[k] color[k] color[k]],linestyle=[:solid :dash :dash],linewidth=[0.25 0.25 0.25],marker=[:xcross :none :none],markersize=[2 0 0])
            plot!(graphTZoomEvolution,t,[zT zT-1.96.*zcT zT+1.96.*zcT],xlims=(0.07*St,0.14*St),ylims=(TZoominf,TZoomsup),yticks=TZoominf:dTZoom:TZoomsup,yflip=false,title="Temperature Profile",xlabel="Time [s]",ylabel="Temperature [K]",label=[typeNames[id] "" ""],color=[color[k] color[k] color[k]],linestyle=[:solid :dash :dash],linewidth=[0.25 0.25 0.25],marker=[:xcross :none :none],markersize=[2 0 0])
            plot!(graphQEvolution,t,[zQ zQ-1.96.*zcQ zQ+1.96.*zcQ],xlims=(0,St),ylims=(Qinf,Qsup),yticks=Qinf:dQ:Qsup,yflip=false,title="Heat Flux Profile",xlabel="Time [s]",ylabel="Heat Flux [W/m" * L"^2" *"]",label=[typeNames[id] "" ""],color=[color[k] color[k] color[k]],linestyle=[:solid :dash :dash],linewidth=[0.25 0.25 0.25],marker=[:xcross :none :none],markersize=[2 0 0])
            plot!(graphTResidueEvolution,t,zTs-zT,xlims=(0,St),ylims=(rTinf,rTsup),yflip=false,title="Temperature's Residue Profile",xlabel="Time [s]",ylabel="Temperature [K]",label=typeNames[id],color=color[k],linestyle=:solid,marker=:xcross,markersize=2)
            plot!(graphQResidueEvolution,t,zQs-zQ,xlims=(0,St),ylims=(rQinf,rQsup),yflip=false,title="Heat Flux's Residue Profile",xlabel="Time [s]",ylabel="Heat Flux [W/m" * L"^2" *"]",label=typeNames[id],color=color[k],linestyle=:solid,marker=:xcross,markersize=2)
        end
    end
    savefig(graphTEvolution,joinpath(imagePath,"TemperatureEvolution.pdf"))
    savefig(graphTZoomEvolution,joinpath(imagePath,"TemperatureZoomEvolution.pdf"))
    savefig(graphQEvolution,joinpath(imagePath,"HeatFluxEvolution.pdf"))
    savefig(graphTsEvolution,joinpath(imagePath,"TemperatureSyntheticEvolution.pdf"))
    savefig(graphQsEvolution,joinpath(imagePath,"HeatFluxSyntheticEvolution.pdf"))
    savefig(graphTResidueEvolution,joinpath(imagePath,"TemperatureResidueEvolution.pdf"))
    savefig(graphQResidueEvolution,joinpath(imagePath,"HeatFluxResidueEvolution.pdf"))
    
    return
end