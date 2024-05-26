MODELS=("TabAutoDiff" "StasyAutoDiff" "CTGAN" "CTABGAN" "TabDDPM" "Stasy")
RANDOM_STATE=0
N_JOB=1

# Loop through each model and run the python script
for model in "${MODELS[@]}"; do
    echo "Training model: ${model}"
    python notebooks/sample_synthetic_data.py --model ${model} --n_job ${N_JOB} --random_state ${RANDOM_STATE}
done
