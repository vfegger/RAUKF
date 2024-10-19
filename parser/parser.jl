using Pkg;
Pkg.activate(@__DIR__);
Pkg.instantiate();
cd(@__DIR__);

using FileIO
using Interpolations

# Function to parse CSV and write to binary
function parse_and_convert_to_binary(input_csv::String, output_bin::String, case_points)
    # Read the CSV files
    data = split.(readlines(input_csv), ",")
    numbers = hcat([parse.(Float64, row) for row in data]...)
    # Celsius to Kelvin
    numbers .+= 273.15
    itp = LinearInterpolation((1:640, 1:480), numbers)
    values = [itp(p...) for p in case_points]

    # Open the binary file for writing
    open(output_bin, "w") do io
        write(io, values)
    end
end

# Example usage
function importfiles(path_input, path_output, name_input, name_output, start, stop, stride, case_points)
    mkpath(path_output)
    for (i, j) in enumerate(start:stride:stop)
        input_csv = path_input * name_input * string(j) * ".csv"
        output_bin = path_output * name_output * string(i - 1) * ".bin"
        parse_and_convert_to_binary(input_csv, output_bin, case_points)
    end
end
input_path = "../measurements/"
output_path = "../input/"

case2_corners = tuple((166, 126), (385, 126), (385, 345), (166, 345))
case3_corners = tuple((173, 119), (401, 126), (396, 354), (166, 347))
Lx = 32;
Ly = 32;
case2_points = collect([case2_corners[1] .+ (case2_corners[2] .- case2_corners[1]) .* (j + 0.5) ./ Lx .+ (case2_corners[4] .- case2_corners[1]) .* (i + 0.5) ./ Ly for i in 0:Lx-1, j in 0:Ly-1])
case3_points = collect([case3_corners[1] .+ (case3_corners[2] .- case3_corners[1]) .* (j + 0.5) ./ Lx .+ (case3_corners[4] .- case3_corners[1]) .* (i + 0.5) ./ Ly for i in 0:Lx-1, j in 0:Ly-1])

Rec2_Time = (36 - 20) * 60 + (39.611 - 52.728)
Rec3_Time = (46 - 25) * 60 + (3.665 - 43.837)
Rec2_i0 = 1999
Rec2_i1 = 4499
Rec3_i0 = 799
Rec3_i1 = 5999

Rec2_L = 28386
Rec3_L = 36635

Rec2_newTime = Rec2_Time * (Rec2_i1 - Rec2_i0 + 1) / Rec2_L
Rec3_newTime = Rec3_Time * (Rec3_i1 - Rec3_i0 + 1) / Rec3_L
println("Case 2: Lt = ", Rec2_i1 - Rec2_i0 + 1, "; St = ", Rec2_newTime, "; Δt = ", Rec2_newTime / (Rec2_i1 - Rec2_i0))
println("Case 3: Lt = ", Rec3_i1 - Rec3_i0 + 1, "; St = ", Rec3_newTime, "; Δt = ", Rec3_newTime / (Rec3_i1 - Rec3_i0))
importfiles(input_path * "case2/", output_path * "case2/", "Rec-0002_", "Values", 1999, 4499, 1, case2_points)
importfiles(input_path * "case3/", output_path * "case3/", "Rec-0003_", "Values", 799, 5999, 1, case3_points)
