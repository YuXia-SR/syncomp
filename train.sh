
# Train TabDDPM related models

MODELS=("TabAutoDiff" "TabDDPM")
RANDOM_STATE=1
N_JOB=5

for model in "${MODELS[@]}"; do
    echo "Training model: ${model}"
    python notebooks/sample_synthetic_data.py --model ${model} --n_job ${N_JOB} --random_state ${RANDOM_STATE}
done


# Train other models 

MODELS=("StasyAutoDiff" "CTGAN" "CTABGAN" "Stasy" "AutoGAN")
N_JOB=1

for model in "${MODELS[@]}"; do
    echo "Training model: ${model}"
    python notebooks/sample_synthetic_data.py --model ${model} --n_job ${N_JOB} --random_state ${RANDOM_STATE}
done