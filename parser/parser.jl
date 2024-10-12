using CSV
using DelimitedFiles

# Custom format function
function custom_format(x)
    # Example: convert to Float32 and multiply by 100
    return Float32(x * 100)
end

# Function to parse CSV and write to binary
function parse_and_convert_to_binary(input_csv::String, output_bin::String)
    # Read the CSV file
    data = CSV.read(input_csv, DataFrame)
    
    # Open the binary file for writing
    open(output_bin, "w") do io
        for row in eachrow(data)
            for val in row
                # Apply custom format function
                formatted_val = custom_format(val)
                # Write the formatted value as binary
                write(io, formatted_val)
            end
        end
    end
end

# Example usage
function importfiles(path_input, path_output, name_input, name_output, start, stop, stride)
    for (i,j) in enumerate(start:stride:stop)
        input_csv = path_input * name_input * string(j) * ".csv"
        output_bin = path_output * name_output * string(i) * ".bin"
        parse_and_convert_to_binary(input_csv, output_bin)
    end
end
input_path = "../measurements/"
output_path = "../input/"
mkdir(input_path)
mkdir(output_path)

importfiles(input_path, output_path, "Measures", "Values", 1, 10, 1)
