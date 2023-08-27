using Cairo
using Gtk
using Plots
using Images
using FileIO

const raukfDataPath = "../data/"
const raukfDataReadyPath = "../data/ready/"

const ioT = PipeBuffer()
const ioQ = PipeBuffer()

Lx = 12
Ly = 12
Lz = 6
Lt = 100

Lxy = Lx*Ly
Lxyz = Lxy*Lz

Lfile = 2*Lx*Ly*(Lz+1)+1

t = zeros(0)
T = zeros(0)
cT = zeros(0)
Q = zeros(0)
cQ = zeros(0)

files = []

windowAlive = [true]

plotT = [Plots.plot()]
plotQ = [Plots.plot()] 

function plotGraph(plot,t,var,cvar,stride,offset)
    plot[] = Plots.plot(t, [var[begin+offset:stride:end] (var[begin+offset:stride:end].+1.96.*sqrt.(cvar[begin+offset:stride:end])) (var[begin+offset:stride:end].-1.96.*sqrt.(cvar[begin+offset:stride:end]))])
end

function checkNewFiles(oldNames)
    names = readdir(raukfDataReadyPath, join=false)
    newFiles = []
    for name in names
        if name ∉ oldNames
            push!(newFiles,name)
        end
    end
    if length(newFiles) > 0
        iStr = sortperm(parse.(Int32,newFiles))
        for index in iStr
            name = newFiles[index]
            data = Array{Float64}(undef,Lfile)
            read!(joinpath(raukfDataPath,string("Values", name, ".bin")), data)
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
            
            push!(oldNames,name)
        end
        return true
    end
    return false
end

function cleanLists(t,T,cT,Q,cQ,files)
    empty!(t)
    empty!(T)
    empty!(cT)
    empty!(Q)
    empty!(cQ)
    empty!(files)
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
        show(ioT, MIME("image/png"), plotT[])
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
        show(ioQ, MIME("image/png"), plotQ[])
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
        if checkNewFiles(files)
            plotGraph(plotT,t,T,cT,Lxyz,Int(Lxy/2+Lx/2))
            plotGraph(plotQ,t,Q,cQ,Lxy,Int(Lxy/2+Lx/2))
        end
        draw(canT)
        draw(canQ)
    end
    id = signal_connect(buttonClean,"clicked") do widget
        cleanLists(t,T,cT,Q,cQ,files)
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
        cleanLists(t,T,cT,Q,cQ,files)
    end
end