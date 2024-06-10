#!/bin/bash

# Command line arguments with defaults
input_dir="${1:-None}"
judge_model_name="$2"
output_dir="${3:-None}"

# Check if input directory is not 'None' and copy data from it
if [ "$input_dir" != "None" ]; then
    cp -r "$input_dir"/arena-hard-v0.1 data/
fi

# Run judgment generation
python gen_judgment.py --setting-file config/judge_config_test.yaml --endpoint-file config/api_config_test.yaml

# Show results
python show_result.py --judge-name "$judge_model_name"

# Copy results to output dir if not 'None'
if [ "$output_dir" != "None" ]; then
    cp -r data/arena-hard-v0.1 "$output_dir"
fi
