#!/bin/bash

# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=10000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
   $2 < threshold { print $1; exit }
  '
}

# Initial port number
port=6041

input_dir="data/gs_datasets"
output_dir="output"
log_dir="logs_prune"

# Only one dataset specified here, but you could run multiple
declare -a run_args=(
    # Mip-Nerf360
    "bicycle"
    "bonsai"
    "counter"
    "flowers"
    "garden"
    "kitchen"
    "room"
    "stump"
    "treehill"
    # Tanks & Temples
    "train"
    "truck"
    # Deep Blending
    "drjohnson"
    "playroom"
  )

# In the paper, we set lambda_mask to 0.15 for Mip-Nerf360 and 0.1 for Tanks & Temples and Deep Blending  
# to ensure a fair comparison with LightGaussian by matching the approximate number of Gaussians.  
# However, 0.1 is generally a robust choice. 
declare -a lambda_mask=(0.1)  


# Loop over the arguments array
for arg in "${run_args[@]}"; do
  # Wait for an available GPU
  while true; do
    gpu_id=$(get_available_gpu)
    if [[ -n $gpu_id ]]; then
      echo "GPU $gpu_id is available. Starting prune_finetune.py with dataset '$arg', lambda_mask '$lambda_mask' on port $port"
      
      CUDA_VISIBLE_DEVICES=$gpu_id python prune_finetune.py \
        -s "$input_dir/$arg" \
        -m "$output_dir/${arg}_${lambda_mask}" \
        --eval \
        --port $port \
        --start_checkpoint "YOUR/PATH/TO/CHECKPOINT/${arg}/chkpnt30000.pth" \
        --iteration 35000 \
        --lambda_mask $lambda_mask \
        --position_lr_max_steps 35000 >> "logs_prune/${arg}_lam${lambda_mask}_prunned.log" 2>&1 
      
      CUDA_VISIBLE_DEVICES=$gpu_id python render.py -m "$output_dir/${arg}_${lambda_mask}" --skip_train >> "$log_dir/${arg}_lam${lambda_mask}_prunned.log" 2>&1 
      CUDA_VISIBLE_DEVICES=$gpu_id python metrics.py -m "$output_dir/${arg}_${lambda_mask}" >> "$log_dir/${arg}_lam${lambda_mask}_prunned.log" 2>&1 

      # Increment the port number for the next run
      ((port++))
      # Allow some time for the process to initialize and potentially use GPU memory
      sleep 60
      break
    else
      echo "No GPU available at the moment. Retrying in 1 minute."
      sleep 60
    fi
  done
done
wait
echo "All prune_finetune.py runs completed."
