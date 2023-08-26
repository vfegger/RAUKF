using Cairo
using Gtk
using Plots
using Images
using FileIO

const raukfDataPath = "../data/"
const raukfDataReadyPath = "../data/ready/"

ioT = PipeBuffer()
ioQ = PipeBuffer()

Lx = 24
Ly = 24
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

windowAlive = true

function plotGraphT(t,T,cT,stride)
    plotT = plot(t, [T[begin:stride:end] (T[begin:stride:end].+cT[begin:stride:end]) (T[begin:stride:end].-cT[begin:stride:end])])
    show(ioT, MIME("image/png"), plotT)
end

function plotGraphQ(t,Q,cQ,stride)
    plotQ = plot(t, [Q[begin:stride:end] (Q[begin:stride:end].+cQ[begin:stride:end]) (Q[begin:stride:end].-cQ[begin:stride:end])])
    show(ioQ, MIME("image/png"), plotQ)
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
        empty!(newFiles)
        return true
    end
    return false
end

function plotCanvas(h = 900, w = 800, type = :v)
    win = GtkWindow("Normal Histogram Widget", h, w) |> (box = GtkBox(type))
    set_gtk_property!(box,:spacing,10)
    canT = GtkCanvas()
    canQ = GtkCanvas()
    push!(box, canT)
    push!(box, canQ)
    set_gtk_property!(box, :expand, canT, true)
    set_gtk_property!(box, :expand, canQ, true)
    @guarded draw(canT) do widget
        ctx = getgc(canT)
        checkNewFiles(files)
        plotGraphT(t,T,cT,Lxyz)
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
        checkNewFiles(files)
        plotGraphQ(t,Q,cQ,Lxy)
        imgQ = read_from_png(ioQ)
        Cairo.save(ctx)
        s = min(width(canQ)/width(imgQ),height(canQ)/height(imgQ))
        scale(ctx,s,s)
        set_source_surface(ctx, imgQ, 0, 0)
        paint(ctx)
        Cairo.restore(ctx)
    end
    id = signal_connect(win,"destroy") do widget
        global windowAlive = false
    end
    showall(win)
    show(canT)
    show(canQ)
    return [win canT canQ]
end

function startMonitor()
    global windowAlive = true
    elements = plotCanvas()

    while windowAlive == true
        reveal(elements[2],true)
        reveal(elements[3],true)
        sleep(3)
    end
end

function cleanMonitor(t, T, cT, Q, cQ, files)
    t = zeros(0)
    T = zeros(0)
    cT = zeros(0)
    Q = zeros(0)
    cQ = zeros(0)
    
    files = []
    println("All data cleaned.")
end