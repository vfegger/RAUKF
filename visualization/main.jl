using Cairo
using Gtk
using Plots
using Images
using FileIO

const ioT = PipeBuffer()
const ioQ = PipeBuffer()

const dataPath = joinpath("..","data")

const types = [:kf, :kfaem, :raukf]
const typePaths = Dict(id => String(id) for id in types)
const typeParms = [:Lx, :Ly, :Lz, :Lt, :Lxy, :Lxyz, :Lfile]
const typeValues = [:t, :T, :cT, :Q, :cQ, :Tm, :cTm, :Ts, :Qs]

const graphTypes = [:T, :Q]

windowAlive = [true]

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
    dataParms[id][7] =  2 * aux[1] * aux[2] * aux[3] + 6 * aux[1] * aux[2] + 1 # Lfile
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
        elseif s == :Ts || s == :Qs
            temp = dataParms[id][5]
        else 
            println("Symbol ",s," Not defined")
        end
        dataOffset[id][s] = offset
        dataSizes[id][s] = temp
        offset = offset + temp
    end
end


println(dataParms)

files_ref = Dict(id => Array{String,1}() for id in types)

data = Dict(id => Dict(idd => zeros(1,5) for idd in types) for id in graphTypes)

graph = Dict(id => Plots.plot() for id in graphTypes)

function combineGraphs(graph, data, graphTypes, types)
    for gt in graphTypes
        col = palette([:blue, :red],size(types,1))
        dgt = data[gt]
        ggt = graph[gt]
        for (i,id) in enumerate(types)
            dgtid = dgt[id]
            numser = size(dgtid,2) - 1
            if numser <= 0
                println("Id: ",id,"; Values: ", dgtid)
                return
            end
            
            lb = Array{String,2}(undef,1,numser)
            lb[1,1] = "Reference: " * typePaths[id]
            lb[1,2] = typePaths[id]
            lb[1,3:end] .= ""

            ls =  Array{Symbol,2}(undef,1,numser)
            ls[1,1] = :dot
            ls[1,2] = :solid
            ls[1,3:end] .= :dash

            plot!(ggt,dgtid[begin:end,1],dgtid[begin:end,2:end],label=lb, color=col[i],linestyle=ls)
        end
    end
end

function dataGraph(t,rvar,var,cvar,stride,offset)
    return [t rvar[begin+offset:stride:end] var[begin+offset:stride:end] (var[begin+offset:stride:end].+1.96.*sqrt.(cvar[begin+offset:stride:end])) (var[begin+offset:stride:end].-1.96.*sqrt.(cvar[begin+offset:stride:end]))]
end

function checkNewFiles(id,oldNames,parms,values,valuesSizes,valuesOffsets,typeValues,mainPath,typePaths)
    hasNewFiles = false
    path = joinpath(mainPath, typePaths[id])
    names = readdir(joinpath(path,"ready"), join=false)
    newFiles = []
    for name in names
        if name ∉ oldNames[id]
            push!(newFiles,name)
        end
    end
    if length(newFiles) > 0
        iStr = sortperm(parse.(Int32,newFiles))
        println(path," ",parms[id][7])
        for index in iStr
            name = newFiles[index]
            data = Array{Float64}(undef,parms[id][7])
            read!(joinpath(path,string("Values", name, ".bin")), data)
            for (i,s) in enumerate(typeValues)
                offset = valuesOffsets[id][s]
                len = valuesSizes[id][s]
                append!(values[id][s],data[offset+1:offset+len])
            end
        end
        append!(oldNames[id],newFiles)
        hasNewFiles = true
    end
    return hasNewFiles
end

function cleanLists(types,typeValues,dataValues,files)
    for id in types
        for s in typeValues
            empty!(dataValues[id][s])
        end
        empty!(files[id])
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
        plot!(graph[:T],size=[width(canT) height(canT)])
        show(ioT, MIME("image/png"), graph[:T])
        imgT = read_from_png(ioT)
        Cairo.save(ctx)
        set_source_surface(ctx, imgT, 0, 0)
        paint(ctx)
        Cairo.restore(ctx)
    end
    @guarded draw(canQ) do widget
        ctx = getgc(canQ)
        plot!(graph[:Q],size=[width(canQ) height(canQ)])
        show(ioQ, MIME("image/png"), graph[:Q])
        imgQ = read_from_png(ioQ)
        Cairo.save(ctx)
        set_source_surface(ctx, imgQ, 0, 0)
        paint(ctx)
        Cairo.restore(ctx)
    end
    id = signal_connect(win,"destroy") do widget
        windowAlive[] = false
    end
    id = signal_connect(buttonRefresh,"clicked") do widget
        recreateGraphs = false
        for id in types
            if checkNewFiles(id,files_ref,dataParms,dataValues,dataSizes,dataOffset,typeValues,dataPath,typePaths)
                data[:T][id] = dataGraph(dataValues[id][:t],dataValues[id][:Ts],dataValues[id][:Tm],dataValues[id][:cTm],dataParms[id][5],Int((dataParms[id][5] + dataParms[id][1]) / 2.0))
                data[:Q][id] = dataGraph(dataValues[id][:t],dataValues[id][:Qs],dataValues[id][:Q],dataValues[id][:cQ],dataParms[id][5],Int((dataParms[id][5] + dataParms[id][1]) / 2.0))
                recreateGraphs = true
            end
        end
        if recreateGraphs
            combineGraphs(graph,data,graphTypes,types)
        end
        draw(canT)
        draw(canQ)
    end
    id = signal_connect(buttonClean,"clicked") do widget
        cleanLists(types,typeValues,dataValues,files_ref)
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
        cleanLists(types,typeValues,dataValues,files_ref)
    end
end

function printGraphs()
    cleanLists(types,typeValues,dataValues,files_ref)
    for id in types
        if checkNewFiles(id,files_ref,dataParms,dataValues,dataSizes,dataOffset,typeValues,dataPath,typePaths)
            Lx = dataParms[id][1]
            Ly = dataParms[id][2]
            x = range(0,0.12,Lx)
            y = range(0,0.12,Ly)
            zT = reshape(dataValues[id][:T][end-Lx*Ly+1:end],Lx,Ly)
            zQ = reshape(dataValues[id][:Q][end-Lx*Ly+1:end],Lx,Ly)
            
            temp = surface(x,y,zT)
            temp = surface(x,y,zQ)
            gui(temp)
        end
    end
end