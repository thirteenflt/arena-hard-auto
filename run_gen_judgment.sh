#!/bin/bash

# Command line arguments with defaults
input_dir="${1:-None}"
output_dir="${2:-None}"

# Check if input directory is not 'None' and copy data from it
if [ "$input_dir" != "None" ]; then
    cp -r "$input_dir"/arena-hard-v0.1 data/
fi

# Run judgment generation
python gen_judgment.py --setting-file config/judge_config_test.yaml --endpoint-file config/api_config_test.yaml

# Show results
python show_result.py --judge-name tscience-uks-gpt4-1106

# Copy results to output dir if not 'None'
if [ "$output_dir" != "None" ]; then
    cp -r data/arena-hard-v0.1 "$output_dir"
fi
