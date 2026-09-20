#!/bin/bash

module load miniforge3
module load cuda/12.0
conda activate alpaca-eval

pred_path=${1:-}

export IS_ALPACA_EVAL_2=False

# Make sure to set OPENAI_API_KEY in the environment
echo "IS_ALPACA_EVAL_2 is: $IS_ALPACA_EVAL_2"
echo "The value of pred_path is: $pred_path"

cd ../../
pwd

alpaca_eval --model_outputs ./data/alpaca/${pred_path} \
   --annotators_config alpaca_eval_gpt4 \
   --output_path ./data/alpaca/output_${pred_path}