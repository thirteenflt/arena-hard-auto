#!/bin/bash

# Pass the model name and port as command line arguments to this script
model_name="$1"
port="${2:-8019}"  # Default port value is 8019
question_file="${3:-question.jsonl}" # Default question_file value is 'question.jsonl'
output_dir="${4:-None}"  # Default output_dir value is 'None'

# Start the model vllm hosting
nohup python -m vllm.entrypoints.openai.api_server --model "$model_name" --dtype auto --api-key token-abc123 --port "$port" > server_output.log 2>&1 &

# Wait for the server to start
sleep 30

# Run answer generation 
python gen_answer.py --setting-file config/gen_answer_config_test.yaml --endpoint-file config/api_config_test.yaml --question-file "$question_file"

# Stop the model vllm hosting
pkill -f vllm.entrypoints.openai.api_server

# Wait for the server to stop
sleep 10

# Copy results to output dir if not 'None'
if [ "$output_dir" != "None" ]; then
  cp -r data/arena-hard-v0.1 "$output_dir"
fi