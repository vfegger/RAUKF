#!/bin/bash

# Check if an argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <result_name>"
    exit 1
fi

rm data/kf/*.bin data/kf/ready/*

# Run the command
./build/HFE_RAUKF_test 1 $2 $3 $4 $5

# Create the results directory if it doesn't exist
mkdir -p results
mkdir -p results/$1

# Move the generated data to the results directory with the given name
cp data/kf/* "results/$1"
