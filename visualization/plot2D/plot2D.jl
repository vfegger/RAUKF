using Plots
using LaTeXStrings
using Printf

const dataPath = joinpath("..", "..", "data", "kf")
const imagePath = joinpath(".", "Images")

const typeParms = [:Lx, :Ly, :Lt]
Ls = Array{Int32}(undef, 3)
Ss = Array{Float64}(undef, 3)
read!(joinpath(dataPath, string("Parms.bin")), Ls)
Ss .= [2.96e-2, 2.96e-2, 1e3]
Lx = Ls[1];
Ly = Ls[2];
Lt = Ls[3];
Sx = Ss[1];
Sy = Ss[2];
St = Ss[3];

const typeValues = [:t, :T, :cT, :Q, :cQ, :Tm, :cTm, :TsN, :Ts, :Qs]
const typeSizes = Dict(
    :t => (1, (1, 1)),
    :T => (Lx * Ly, (Lx, Ly)),
    :cT => (Lx * Ly, (Lx, Ly)),
    :Q => (Lx * Ly, (Lx, Ly)),
    :cQ => (Lx * Ly, (Lx, Ly)),
    :Tm => (Lx * Ly, (Lx, Ly)),
    :cTm => (Lx * Ly, (Lx, Ly)),
    :TsN => (Lx * Ly, (Lx, Ly)),
    :Ts => (Lx * Ly, (Lx, Ly)),
    :Qs => (Lx * Ly, (Lx, Ly)),
)
#[t T cT Q cQ Tm cTm Ts Qs]
dataValues = Dict(id => Array{Float64,3}(undef, last(typeSizes[id])[1], last(typeSizes[id])[2], Lt) for id in typeValues)

function getData()
    names = readdir(joinpath(dataPath, "ready"), join=false)
    if length(names) > 0
        sortedNames = sortperm(parse.(Int32, names))
        for name in names[sortedNames]
            println("Reading data from file: ", name)
            index = parse(Int32, name) + 1
            if index > Lt
                continue
            end
            name = names[index]
            data = Array{Float64}(undef, sum(first.(values(typeSizes))))
            read!(joinpath(dataPath, "Values" * name * ".bin"), data)
            offset = 0
            for type in typeValues
                L = first(typeSizes[type])
                typeSize = last(typeSizes[type])
                dataValues[type][:, :, index] .= reshape(data[1+offset:L+offset], typeSize)
                offset += L
            end
        end
    end
end

function printProfiles(case, t)
    @assert t >= 1
    @assert t <= Lt
    dx2 = Sx / Lx / 2
    dy2 = Sy / Ly / 2
    amp = 1e3

    x = range(0 + dx2, Sx - dx2, Lx)
    y = range(0 + dy2, Sy - dy2, Ly)
    zT = @view(dataValues[:T][:, :, t])
    zcT = @view(dataValues[:cT][:, :, t])
    zQ = @view(dataValues[:Q][:, :, t])
    zcQ = @view(dataValues[:cQ][:, :, t])
    zTm = @view(dataValues[:Tm][:, :, t])
    zcTm = @view(dataValues[:cTm][:, :, t])
    zTsN = @view(dataValues[:TsN][:, :, t])
    zTs = @view(dataValues[:Ts][:, :, t])
    zQs = @view(dataValues[:Qs][:, :, t])

    colgrad = cgrad(:thermal, rev=false)

    mkpath(joinpath(imagePath, case))

    plt_T_Profile = heatmap(x, y, zT, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Temperature", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_T_Profile, joinpath(imagePath, case, "TemperatureProfile_" * string(t) * ".pdf"))

    plt_cT_Profile = heatmap(x, y, zcT, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_cT_Profile, joinpath(imagePath, case, "TemperatureCovarianceProfile_" * string(t) * ".pdf"))

    plt_Q_Profile = heatmap(x, y, zQ, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Q_Profile, joinpath(imagePath, case, "HeatFluxProfile_" * string(t) * ".pdf"))

    plt_cQ_Profile = heatmap(x, y, zcQ, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_cQ_Profile, joinpath(imagePath, case, "HeatFluxCovarianceProfile_" * string(t) * ".pdf"))

    plt_Tm_Profile = heatmap(x, y, zTm, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Tm_Profile, joinpath(imagePath, case, "ObservedTemperatureProfile_" * string(t) * ".pdf"))

    plt_cTm_Profile = heatmap(x, y, zcTm, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_cTm_Profile, joinpath(imagePath, case, "ObservedTemperatureCovarianceProfile_" * string(t) * ".pdf"))

    plt_TsN_Profile = heatmap(x, y, zTsN, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_TsN_Profile, joinpath(imagePath, case, "NoisyMeasureTemperatureProfile_" * string(t) * ".pdf"))

    plt_Ts_Profile = heatmap(x, y, zTs, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Ts_Profile, joinpath(imagePath, case, "MeasureTemperatureProfile_" * string(t) * ".pdf"))

    plt_Qs_Profile = heatmap(x, y, zQs, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Qs_Profile, joinpath(imagePath, case, "SimulatedHeatFluxProfile_" * string(t) * ".pdf"))

    display(plt_T_Profile)
end

function printEvolutions(case, x, y)
    
end


getData()