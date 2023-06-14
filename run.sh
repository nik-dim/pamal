#!/bin/bash
# Utility script to launch experiments with different hyperparameters. All seeds are launched in the same GPU. 
# The script uses the Pueue task manager to launch the experiments in parallel. 
# Link: https://github.com/Nukesor/pueue

# An equivalent way of launching experiments is via the multirun functionality of hydra. However, this has three problems 
# that are solved by this script:
# 1. distributing the experiments over multiple GPUs
# 2. following the specific run via `pueue follow`. Otherwise, the terminal output is massive and and hard to follow.
# 3. the "script" value in the Wandb dashboard is not set properly.

# ==================================================================================================

# change the path to the script
CMD="utkface.py"

seeds=(0 1 2)
gpu_ids=(2 3)

# Define the ranges for each hyperparameter
methods=(autol)

# Calculate the total number of combinations
total_combinations=$(( 1 * ${#methods[@]} ))

# Iterate over all possible combinations of hyperparameters
for (( combination_idx=0; combination_idx<total_combinations; combination_idx++ )); do
    # Calculate the hyperparameter values for this combination
    method_idx=$(( combination_idx % ${#methods[@]} ))
    method=${methods[$method_idx]}

    # Calculate the GPU index for this combination
    gpu_idx=$(( combination_idx % ${#gpu_ids[@]} ))
    gpu_idx=${gpu_ids[$gpu_idx]}  

    for seed in "${seeds[@]}"; do
        gpu_id=${gpu_ids[$gpu_idx]}

        # Generate a unique identifier for the combination
        command="CUDA_VISIBLE_DEVICES=${gpu_idx} pueue add -g gpu${gpu_idx} python ${CMD} method=${method} seed=${seed}"
        echo "$command"

        # Launch the experiment
        CUDA_VISIBLE_DEVICES=${gpu_idx} pueue add -g gpu${gpu_idx} python ${CMD} method=${method} seed=${seed}
    done
    echo ""
done



# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 