using Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate();
cd(@__DIR__);

using Plots
using LaTeXStrings
using Printf
using Interpolations

const amp = 5e3
const isExperimental = parse(Int32, ARGS[2]) != 0

const dataPath = joinpath("..", "..", "results", ARGS[1])
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

plot_font = "Computer Modern"
default(fontfamily=plot_font)

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

function getData(dataPath)
    names = readdir(joinpath(dataPath, "ready"), join=false)
    if length(names) > 0
        sortedNames = sortperm(parse.(Int32, names))
        for name in names[sortedNames]
            index = parse(Int32, name) + 1
            if index > Lt
                continue
            end
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

function printProfiles_IP(case, t)
    @assert t >= 1
    @assert t <= Lt
    dx2 = Sx / Lx / 2
    dy2 = Sy / Ly / 2
    X = range(0 + dx2, Sx - dx2, Lx)
    Y = range(0 + dx2, Sy - dy2, Ly)
    t0 = first(dataValues[:t])
    t1 = last(dataValues[:t])
    T = range(t0, t1, Lt)
    Ω = (X, Y, T)
    itp_T = LinearInterpolation(Ω, dataValues[:T])
    itp_cT = LinearInterpolation(Ω, sqrt.(dataValues[:cT]))
    itp_Q = LinearInterpolation(Ω, dataValues[:Q])
    itp_cQ = LinearInterpolation(Ω, sqrt.(dataValues[:cQ]))
    itp_Tm = LinearInterpolation(Ω, dataValues[:Tm])
    itp_cTm = LinearInterpolation(Ω, sqrt.(dataValues[:cTm]))
    itp_TsN = LinearInterpolation(Ω, dataValues[:TsN])
    itp_Ts = LinearInterpolation(Ω, dataValues[:Ts])
    itp_Qs = LinearInterpolation(Ω, dataValues[:Qs])

    points = [(x, y, t) for x in X for y in Y]

    zT = reshape([itp_T(p...) for p in points], (Lx, Ly))
    zcT = reshape([itp_cT(p...) for p in points], (Lx, Ly))
    zQ = reshape([itp_Q(p...) for p in points], (Lx, Ly))
    zcQ = reshape([itp_cQ(p...) for p in points], (Lx, Ly))
    zTm = reshape([itp_Tm(p...) for p in points], (Lx, Ly))
    zcTm = reshape([itp_cTm(p...) for p in points], (Lx, Ly))
    zTsN = reshape([itp_TsN(p...) for p in points], (Lx, Ly))
    zTs = reshape([itp_Ts(p...) for p in points], (Lx, Ly))
    zQs = reshape([itp_Qs(p...) for p in points], (Lx, Ly))

    colgrad = cgrad(:thermal, rev=false)

    mkpath(joinpath(imagePath, case))

    T_min = (floor(min(minimum(zT))) - 4) ÷ 5 * 5
    T_max = (ceil(max(maximum(zT))) + 4) ÷ 5 * 5

    Tm_min = (floor(min(minimum(zTm), minimum(zTsN), minimum(zTs))) - 4) ÷ 5 * 5
    Tm_max = (ceil(max(maximum(zTm), maximum(zTsN), maximum(zTs))) + 4) ÷ 5 * 5

    Q_min = (floor(min(minimum(zQ), minimum(zQs))) - 4) ÷ 5 * 5
    Q_max = (ceil(max(maximum(zQ), maximum(zQs))) + 4) ÷ 5 * 5

    R_min = (floor(min(minimum(zTm .- zTsN))) - 4) ÷ 5 * 5
    R_max = (ceil(max(maximum(zTm .- zTsN))) + 4) ÷ 5 * 5

    plt_T_Profile = heatmap(X, Y, zT, xlims=(0, Sx), ylims=(0, Sy), clims=(T_min, T_max), yflip=false, c=colgrad, aspect_ratio=:equal, title="Temperature", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_T_Profile, joinpath(imagePath, case, "TemperatureProfile_" * string(t) * ".pdf"))

    plt_cT_Profile = heatmap(X, Y, zcT, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Temperature Standard Deviation", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature Standard Deviation [K]", dpi=1000)
    savefig(plt_cT_Profile, joinpath(imagePath, case, "TemperatureCovarianceProfile_" * string(t) * ".pdf"))

    plt_Q_Profile = heatmap(X, Y, zQ, xlims=(0, Sx), ylims=(0, Sy), clims=(Q_min, Q_max), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Q_Profile, joinpath(imagePath, case, "HeatFluxProfile_" * string(t) * ".pdf"))

    plt_cQ_Profile = heatmap(X, Y, zcQ, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Heat Flux Standard Deviation", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux Standard Deviation [W/m^2]", dpi=1000)
    savefig(plt_cQ_Profile, joinpath(imagePath, case, "HeatFluxCovarianceProfile_" * string(t) * ".pdf"))

    plt_Tm_Profile = heatmap(X, Y, zTm, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Observed Temperature", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_Tm_Profile, joinpath(imagePath, case, "ObservedTemperatureProfile_" * string(t) * ".pdf"))

    plt_cTm_Profile = heatmap(X, Y, zcTm, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Observed Temperature Deviation", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature Standard Deviation [K]", dpi=1000)
    savefig(plt_cTm_Profile, joinpath(imagePath, case, "ObservedTemperatureCovarianceProfile_" * string(t) * ".pdf"))

    plt_TsN_Profile = heatmap(X, Y, zTsN, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Measured Temperature", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_TsN_Profile, joinpath(imagePath, case, "NoisyMeasureTemperatureProfile_" * string(t) * ".pdf"))

    plt_Ts_Profile = heatmap(X, Y, zTs, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Measured Temperature", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_Ts_Profile, joinpath(imagePath, case, "MeasureTemperatureProfile_" * string(t) * ".pdf"))

    plt_Qs_Profile = heatmap(X, Y, zQs, xlims=(0, Sx), ylims=(0, Sy), yflip=false, c=colgrad, aspect_ratio=:equal, title="Reference Heat Flux", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Heat Flux [W/m^2]", dpi=1000)
    savefig(plt_Qs_Profile, joinpath(imagePath, case, "SimulatedHeatFluxProfile_" * string(t) * ".pdf"))

    plt_rT_Profile = heatmap(X, Y, zTm .- zTsN, xlims=(0, Sx), ylims=(0, Sy), clims=(R_min, R_max), yflip=false, c=colgrad, aspect_ratio=:equal, title="Temperature Residual", xlabel="X [m]", ylabel="Y [m]", colorbar_title="Temperature [K]", dpi=1000)
    savefig(plt_rT_Profile, joinpath(imagePath, case, "ResidualTemperatureProfile_" * string(t) * ".pdf"))

    return
end

function printPartialProfile_IP(case, t, x, y, isExperimental)
    @assert t >= 1
    @assert t <= Lt
    dx2 = Sx / Lx / 2
    dy2 = Sy / Ly / 2
    X = range(0 + dx2, Sx - dx2, Lx)
    Y = range(0 + dx2, Sy - dy2, Ly)
    t0 = first(dataValues[:t])
    t1 = last(dataValues[:t])
    T = range(t0, t1, Lt)
    Ω = (X, Y, T)
    itp_T = LinearInterpolation(Ω, dataValues[:T])
    itp_cT = LinearInterpolation(Ω, sqrt.(dataValues[:cT]))
    itp_Q = LinearInterpolation(Ω, dataValues[:Q])
    itp_cQ = LinearInterpolation(Ω, sqrt.(dataValues[:cQ]))
    itp_Tm = LinearInterpolation(Ω, dataValues[:Tm])
    itp_cTm = LinearInterpolation(Ω, sqrt.(dataValues[:cTm]))
    itp_TsN = LinearInterpolation(Ω, dataValues[:TsN])
    itp_Ts = LinearInterpolation(Ω, dataValues[:Ts])
    itp_Qs = LinearInterpolation(Ω, dataValues[:Qs])

    pointsX = [(x, y, t) for x in X]
    pointsY = [(x, y, t) for y in Y]

    zT_X = [itp_T(p...) for p in pointsX]
    zT_Y = [itp_T(p...) for p in pointsY]
    zcT_X = [itp_cT(p...) for p in pointsX]
    zcT_Y = [itp_cT(p...) for p in pointsY]
    zQ_X = [itp_Q(p...) for p in pointsX]
    zQ_Y = [itp_Q(p...) for p in pointsY]
    zcQ_X = [itp_cQ(p...) for p in pointsX]
    zcQ_Y = [itp_cQ(p...) for p in pointsY]
    zTm_X = [itp_Tm(p...) for p in pointsX]
    zTm_Y = [itp_Tm(p...) for p in pointsY]
    zcTm_X = [itp_cTm(p...) for p in pointsX]
    zcTm_Y = [itp_cTm(p...) for p in pointsY]
    zTsN_X = [itp_TsN(p...) for p in pointsX]
    zTsN_Y = [itp_TsN(p...) for p in pointsY]
    zTs_X = [itp_Ts(p...) for p in pointsX]
    zTs_Y = [itp_Ts(p...) for p in pointsY]
    zQs_X = [itp_Qs(p...) for p in pointsX]
    zQs_Y = [itp_Qs(p...) for p in pointsY]

    mkpath(joinpath(imagePath, case))

    T_vals_X = isExperimental ? [zTs_X zTm_X zTm_X .+ 1.96 .* zcTm_X zTm_X .- 1.96 .* zcTm_X] : [zTs_X zTsN_X zTm_X zTm_X .+ 1.96 .* zcTm_X zTm_X .- 1.96 .* zcTm_X]
    Q_vals_X = isExperimental ? [zQ_X zQ_X .+ 1.96 .* zcQ_X zQ_X .- 1.96 .* zcQ_X] : [zQs_X zQ_X zQ_X .+ 1.96 .* zcQ_X zQ_X .- 1.96 .* zcQ_X]
    R_vals_X = [zTm_X .- zTsN_X]

    T_vals_Y = isExperimental ? [zTs_Y zTm_Y zTm_Y .+ 1.96 .* zcTm_Y zTm_Y .- 1.96 .* zcTm_Y] : [zTs_Y zTsN_Y zTm_Y zTm_Y .+ 1.96 .* zcTm_Y zTm_Y .- 1.96 .* zcTm_Y]
    Q_vals_Y = isExperimental ? [zQ_Y zQ_Y .+ 1.96 .* zcQ_Y zQ_Y .- 1.96 .* zcQ_Y] : [zQs_Y zQ_Y zQ_Y .+ 1.96 .* zcQ_Y zQ_Y .- 1.96 .* zcQ_Y]
    R_vals_Y = [zTm_Y .- zTsN_Y]

    labels_sim = Dict(
        :T => ["Simulated" "Noisy" "Estimated" "Confidence Interval 95%" ""],
        :Q => ["Reference" "Estimated" "Confidence Interval 95%" ""],
        :R => "Residual",
    )
    labels_exp = Dict(
        :T => ["Measured" "Estimated" "Confidence Interval 95%" ""],
        :Q => ["Estimated" "Confidence Interval 95%" ""],
        :R => "Residual",
    )
    labels = isExperimental ? labels_exp : labels_sim

    linestyles_sim = Dict(
        :T => [:solid :solid :dash :dashdot :dashdot],
        :Q => [:solid :dash :dashdot :dashdot],
        :R => :solid,
    )
    linestyles_exp = Dict(
        :T => [:solid :dash :dashdot :dashdot],
        :Q => [:solid :dashdot :dashdot],
        :R => :solid,
    )
    linestyles = isExperimental ? linestyles_exp : linestyles_sim
    linewidths_sim = Dict(
        :T => [1.5 1.5 1.5 0.5 0.5],
        :Q => [1.5 1.5 0.5 0.5],
        :R => 0.5,
    )
    linewidths_exp = Dict(
        :T => [1.5 1.5 0.5 0.5],
        :Q => [1.5 0.5 0.5],
        :R => 0.5,
    )
    linewidths = isExperimental ? linewidths_exp : linewidths_sim
    seriescolors_sim = Dict(
        :T => [:black :red :blue :blue :blue],
        :Q => [:black :red :blue :blue],
        :R => :black,
    )
    seriescolors_exp = Dict(
        :T => [:black :red :blue :blue],
        :Q => [:red :blue :blue],
        :R => :black,
    )
    seriescolors = isExperimental ? seriescolors_exp : seriescolors_sim

    plt_T_PartialProfile_X = plot(X, T_vals_X, title="Temperature", xlabel="Position X [m]", ylabel="Temperature [K]", label=labels[:T], ls=linestyles[:T], lw=linewidths[:T], seriescolor=seriescolors[:T], dpi=1000)
    savefig(plt_T_PartialProfile_X, joinpath(imagePath, case, "TemperaturePartialProfile_X_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

    plt_Q_PartialProfile_X = plot(X, Q_vals_X, title="Heat Flux", xlabel="Position X [m]", ylabel="Heat Flux [W/m^2]", label=labels[:Q], ls=linestyles[:Q], lw=linewidths[:Q], seriescolor=seriescolors[:Q], dpi=1000)
    savefig(plt_Q_PartialProfile_X, joinpath(imagePath, case, "HeatFluxPartialProfile_X_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

    plt_rT_PartialProfile_X = plot(X, R_vals_X, title="Temperature", xlabel="Position X [m]", ylabel="Residual [K]", label=labels[:R], ls=linestyles[:R], lw=linewidths[:R], seriescolor=seriescolors[:R], dpi=1000)
    savefig(plt_rT_PartialProfile_X, joinpath(imagePath, case, "TemperatureResidualPartialProfile_X_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

    plt_T_PartialProfile_Y = plot(Y, T_vals_Y, title="Temperature", xlabel="Position Y [m]", ylabel="Temperature [K]", label=labels[:T], ls=linestyles[:T], lw=linewidths[:T], seriescolor=seriescolors[:T], dpi=1000)
    savefig(plt_T_PartialProfile_Y, joinpath(imagePath, case, "TemperaturePartialProfile_Y_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

    plt_Q_PartialProfile_Y = plot(Y, Q_vals_Y, title="Heat Flux", xlabel="Position Y [m]", ylabel="Heat Flux [W/m^2]", label=labels[:Q], ls=linestyles[:Q], lw=linewidths[:Q], seriescolor=seriescolors[:Q], dpi=1000)
    savefig(plt_Q_PartialProfile_Y, joinpath(imagePath, case, "HeatFluxPartialProfile_Y_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

    plt_rT_PartialProfile_Y = plot(Y, R_vals_Y, title="Temperature", xlabel="Position Y [m]", ylabel="Residual [K]", label=labels[:R], ls=linestyles[:R], lw=linewidths[:R], seriescolor=seriescolors[:R], dpi=1000)
    savefig(plt_rT_PartialProfile_Y, joinpath(imagePath, case, "TemperatureResidualPartialProfile_Y_" * string(x) * "_" * string(y) * "_" * string(t) * ".pdf"))

end

function printEvolutions(case, x, y, isExperimental)
    @assert x >= 0.0 && x <= Sx
    @assert y >= 0.0 && y <= Sy

    dx2 = Sx / Lx / 2
    dy2 = Sy / Ly / 2
    X = range(0 + dx2, Sx - dx2, Lx)
    Y = range(0 + dx2, Sy - dy2, Ly)
    t0 = first(dataValues[:t])
    t1 = last(dataValues[:t])
    T = range(t0, t1, Lt)
    Ω = (X, Y, T)
    itp_T = LinearInterpolation(Ω, dataValues[:T])
    itp_cT = LinearInterpolation(Ω, dataValues[:cT])
    itp_Q = LinearInterpolation(Ω, dataValues[:Q])
    itp_cQ = LinearInterpolation(Ω, dataValues[:cQ])
    itp_Tm = LinearInterpolation(Ω, dataValues[:Tm])
    itp_cTm = LinearInterpolation(Ω, dataValues[:cTm])
    itp_TsN = LinearInterpolation(Ω, dataValues[:TsN])
    itp_Ts = LinearInterpolation(Ω, dataValues[:Ts])
    itp_Qs = LinearInterpolation(Ω, dataValues[:Qs])

    points = [(x, y, t) for t in T]

    zT = [itp_T(p...) for p in points]
    zcT = sqrt.([itp_cT(p...) for p in points])
    zQ = [itp_Q(p...) for p in points]
    zcQ = sqrt.([itp_cQ(p...) for p in points])
    zTm = [itp_Tm(p...) for p in points]
    zcTm = sqrt.([itp_cTm(p...) for p in points])
    zTsN = [itp_TsN(p...) for p in points]
    zTs = [itp_Ts(p...) for p in points]
    zQs = [itp_Qs(p...) for p in points]

    T_vals = isExperimental ? [zTs zTm zTm .+ 1.96 .* zcTm zTm .- 1.96 .* zcTm] : [zTs zTsN zTm zTm .+ 1.96 .* zcTm zTm .- 1.96 .* zcTm]
    Q_vals = isExperimental ? [zQ zQ .+ 1.96 .* zcQ zQ .- 1.96 .* zcQ] : [zQs zQ zQ .+ 1.96 .* zcQ zQ .- 1.96 .* zcQ]
    R_vals = [zTm .- zTsN]

    labels_sim = Dict(
        :T => ["Simulated" "Noisy" "Estimated" "Confidence Interval 95%" ""],
        :Q => ["Reference" "Estimated" "Confidence Interval 95%" ""],
        :R => "Residual",
    )
    labels_exp = Dict(
        :T => ["Measured" "Estimated" "Confidence Interval 95%" ""],
        :Q => ["Estimated" "Confidence Interval 95%" ""],
        :R => "Residual",
    )
    labels = isExperimental ? labels_exp : labels_sim

    linestyles_sim = Dict(
        :T => [:solid :solid :dash :dashdot :dashdot],
        :Q => [:solid :dash :dashdot :dashdot],
        :R => :solid,
    )
    linestyles_exp = Dict(
        :T => [:solid :dash :dashdot :dashdot],
        :Q => [:solid :dashdot :dashdot],
        :R => :solid,
    )
    linestyles = isExperimental ? linestyles_exp : linestyles_sim
    linewidths_sim = Dict(
        :T => [1.5 1.5 1.5 0.5 0.5],
        :Q => [1.5 1.5 0.5 0.5],
        :R => 0.5,
    )
    linewidths_exp = Dict(
        :T => [1.5 1.5 0.5 0.5],
        :Q => [1.5 0.5 0.5],
        :R => 0.5,
    )
    linewidths = isExperimental ? linewidths_exp : linewidths_sim
    seriescolors_sim = Dict(
        :T => [:black :red :blue :blue :blue],
        :Q => [:black :red :blue :blue],
        :R => :black,
    )
    seriescolors_exp = Dict(
        :T => [:black :red :blue :blue],
        :Q => [:red :blue :blue],
        :R => :black,
    )
    seriescolors = isExperimental ? seriescolors_exp : seriescolors_sim

    mkpath(joinpath(imagePath, case))

    plt_T_Evolution = plot(T, T_vals, title="Temperature", xlabel="Time [s]", ylabel="Temperature [K]", label=labels[:T], ls=linestyles[:T], lw=linewidths[:T], seriescolor=seriescolors[:T], dpi=1000)
    savefig(plt_T_Evolution, joinpath(imagePath, case, "TemperatureEvolution_" * string(x) * "_" * string(y) * ".pdf"))

    plt_Q_Evolution = plot(T, Q_vals, title="Heat Flux", xlabel="Time [s]", ylabel="Heat Flux [W/m^2]", label=labels[:Q], ls=linestyles[:Q], lw=linewidths[:Q], seriescolor=seriescolors[:Q], dpi=1000)
    savefig(plt_Q_Evolution, joinpath(imagePath, case, "HeatFluxEvolution_" * string(x) * "_" * string(y) * ".pdf"))

    plt_rT_Evolution = plot(T, R_vals, title="Temperature", xlabel="Time [s]", ylabel="Residual [K]", label=labels[:R], ls=linestyles[:R], lw=linewidths[:R], seriescolor=seriescolors[:R], dpi=1000)
    savefig(plt_rT_Evolution, joinpath(imagePath, case, "TemperatureResidualEvolution_" * string(x) * "_" * string(y) * ".pdf"))

    return
end


getData(dataPath)

if isExperimental
    printProfiles_IP(ARGS[1], 10)
    printProfiles_IP(ARGS[1], 30)
    printProfiles_IP(ARGS[1], 50)
    printProfiles_IP(ARGS[1], 70)
    printEvolutions(ARGS[1], 0.004, 0.025, isExperimental)
else
    printProfiles_IP(ARGS[1], 59)
    printPartialProfile_IP(ARGS[1], 59, Sx / 2, Sy / 2, isExperimental)
    printEvolutions(ARGS[1], Sx / 2, Sy / 2, isExperimental)
end