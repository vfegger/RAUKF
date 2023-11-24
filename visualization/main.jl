using Cairo
using Gtk
using Plots
using Images
using FileIO

const dataPath = joinpath("..","data")
const typePaths = ["raukf","kf","kfaem"]

const ioT = PipeBuffer()
const ioQ = PipeBuffer()

# [Lx, Ly, Lz, Lt, Lxy, Lxyz, Lfile]
dataParms = Array{Int,2}(undef,7,size(typePaths,1))

for i = 1:size(typePaths,1)
    data = Array{Int32}(undef,4)
    read!(joinpath(dataPath,typePaths[i],string("Parms.bin")), data)
    dataParms[1:4,i] = data[1:4]
    dataParms[5,i] = data[1] * data[2]
    dataParms[6,i] = data[1] * data[2] * data[3]
    dataParms[7,i] = 2 * data[1] * data[2] * (data[3] + 2) + 1
end

println(dataParms)

#[t T cT Q cQ Tm cTm]
t_ref = Array{Array{Float64,1},1}()
T_ref = Array{Array{Float64,1},1}()
cT_ref = Array{Array{Float64,1},1}()
Q_ref = Array{Array{Float64,1},1}()
cQ_ref = Array{Array{Float64,1},1}()
Tm_ref = Array{Array{Float64,1},1}()
cTm_ref = Array{Array{Float64,1},1}()

files_ref = Array{Array{String,1},1}()

dataT = Array{Array{Float64,2},1}()
dataQ = Array{Array{Float64,2},1}()

for i = 1:size(typePaths,1)
    push!(t_ref,Array{Float64,1}())
    push!(T_ref,Array{Float64,1}())
    push!(cT_ref,Array{Float64,1}())
    push!(Q_ref,Array{Float64,1}())
    push!(cQ_ref,Array{Float64,1}())
    push!(Tm_ref,Array{Float64,1}())
    push!(cTm_ref,Array{Float64,1}())

    push!(files_ref,Array{String,1}())
    
    push!(dataT,Array{Float64,2}(undef,1,3))
    push!(dataQ,Array{Float64,2}(undef,1,3))
    dataT[i][1,1:3] .= 0.0
    dataQ[i][1,1:3] .= 0.0
end

windowAlive = [true]

graph = [Plots.plot(),Plots.plot()]

function combineGraphs(graph,index,data)
    graph[index] = Plots.plot()
    pal = palette([:blue, :red], size(data,1))
    for i in 1:size(data,1)
        numser = size(data[i][begin:end,2:end],2)
        if numser == 0
            println(typePaths[index],"",index,":",size(data,1)," ",i," ",data[i])
        end
        labels = Array{String,2}(undef,1,numser)
        labels[1,1] = typePaths[i]
        labels[1,2:end] .= "" 
        linestyles = Array{Symbol,2}(undef,1,numser)
        linestyles[1,1] = :solid
        linestyles[1,2:end] .= :dash

        plot!(graph[index],data[i][begin:end,1],data[i][begin:end,2:end],label=labels,color=pal[i],linestyle=linestyles)
    end
end

function dataGraph(data,index,t,var,cvar,stride,offset)
    data[index] = [t var[begin+offset:stride:end] (var[begin+offset:stride:end].+1.96.*sqrt.(cvar[begin+offset:stride:end])) (var[begin+offset:stride:end].-1.96.*sqrt.(cvar[begin+offset:stride:end]))]
end

function checkNewFiles(oldNames,dataPath,t,T,cT,Q,cQ,Tm,cTm,Lxy,Lxyz,Lfile)
    names = readdir(joinpath(dataPath,"ready"), join=false)
    newFiles = []
    for name in names
        if name ∉ oldNames
            push!(newFiles,name)
        end
    end
    if length(newFiles) > 0
        iStr = sortperm(parse.(Int32,newFiles))
        println(dataPath," ",Lfile)
        for index in iStr
            name = newFiles[index]
            data = Array{Float64}(undef,Lfile)
            read!(joinpath(dataPath,string("Values", name, ".bin")), data)
            # Time
            offset = 0
            push!(t,data[offset+1])
            # Temperature
            offset = offset + 1
            append!(T,data[offset+1:offset+Lxyz])
            # Temperature Error
            offset = offset + Lxyz
            append!(cT,data[offset+1:offset+Lxyz])
            # Heat Flux
            offset = offset + Lxyz
            append!(Q,data[offset+1:offset+Lxy])
            # Heat Flux Error
            offset = offset + Lxy
            append!(cQ,data[offset+1:offset+Lxy])
            # Temperature Measured
            offset = offset + Lxy
            append!(Tm,data[offset+1:offset+Lxy])
            # Temperature Measured Error
            offset = offset + Lxy
            append!(cTm,data[offset+1:offset+Lxy])

            push!(oldNames,name)
        end
        return true
    end
    return false
end

function cleanLists(t,T,cT,Q,cQ,Tm,cTm,files)
    for i = 1:size(t,1)
        empty!(t[i])
        empty!(T[i])
        empty!(cT[i])
        empty!(Q[i])
        empty!(cQ[i])
        empty!(Tm[i])
        empty!(cTm[i])
        empty!(files[i])
    end
end

function plotCanvas(h = 1000, w = 600, type = :v)
    win = GtkWindow("Data Monitor", h, w) |> (box = GtkBox(type))
    set_gtk_property!(box,:spacing,10)
    canT = GtkCanvas()
    canQ = GtkCanvas()
    buttonRefresh = GtkButton()
    buttonClean = GtkButton()
    push!(box, canT)
    push!(box, canQ)
    push!(box, buttonRefresh)
    push!(box, buttonClean)
    set_gtk_property!(box, :expand, canT, true)
    set_gtk_property!(box, :expand, canQ, true)
    set_gtk_property!(buttonRefresh, :label, "Refresh")
    set_gtk_property!(buttonClean, :label, "Clean")
    @guarded draw(canT) do widget
        ctx = getgc(canT)
        show(ioT, MIME("image/png"), graph[1])
        imgT = read_from_png(ioT)
        Cairo.save(ctx)
        s = min(width(canT)/width(imgT),height(canT)/height(imgT))
        scale(ctx,s,s)
        set_source_surface(ctx, imgT, 0, 0)
        paint(ctx)
        Cairo.restore(ctx)
    end
    @guarded draw(canQ) do widget
        ctx = getgc(canQ)
        show(ioQ, MIME("image/png"), graph[2])
        imgQ = read_from_png(ioQ)
        Cairo.save(ctx)
        s = min(width(canQ)/width(imgQ),height(canQ)/height(imgQ))
        scale(ctx,s,s)
        set_source_surface(ctx, imgQ, 0, 0)
        paint(ctx)
        Cairo.restore(ctx)
    end
    id = signal_connect(win,"destroy") do widget
        windowAlive[] = false
    end
    id = signal_connect(buttonRefresh,"clicked") do widget
        recreateGraphs = false
        for i = 1:size(typePaths,1)
            if checkNewFiles(files_ref[i],joinpath(dataPath,typePaths[i]),t_ref[i],T_ref[i],cT_ref[i],Q_ref[i],cQ_ref[i],Tm_ref[i],cTm_ref[i],dataParms[5,i],dataParms[6,i],dataParms[7,i])
                dataGraph(dataT,i,t_ref[i],Tm_ref[i],cTm_ref[i],dataParms[5,i],Int(dataParms[5,i]/2+dataParms[1,i]/2))
                dataGraph(dataQ,i,t_ref[i],Q_ref[i],cQ_ref[i],dataParms[5,i],Int(dataParms[5,i]/2+dataParms[1,i]/2))
                recreateGraphs = true
            end
        end
        if recreateGraphs
            combineGraphs(graph,1,dataT)
            combineGraphs(graph,2,dataQ)
        end
        draw(canT)
        draw(canQ)
    end
    id = signal_connect(buttonClean,"clicked") do widget
        cleanLists(t_ref,T_ref,cT_ref,Q_ref,cQ_ref,Tm_ref,cTm_ref,files_ref)
    end
    showall(win)
    show(canT)
    show(canQ)
    show(buttonRefresh)
    show(buttonClean)
end

function startMonitor()
    windowAlive[] = true
    plotCanvas()

    if !isinteractive()
        c = Condition()
        signal_connect(win, :destroy) do widget
            notify(c)
        end
        @async Gtk.gtk_main()
        wait(c)
        cleanLists(t_ref,T_ref,cT_ref,Q_ref,cQ_ref,Tm_ref,cTm_ref,files_ref)
    end
end