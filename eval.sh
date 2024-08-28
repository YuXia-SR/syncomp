
MODELS=("TabDDPM" "TabAutoDiff" "StasyAutoDiff" "CTGAN" "CTABGAN" "Stasy" "AutoGAN")
RANDOM_STATE=0
SAMPLE_SIZE=10
compute_fidelity=True
compute_utility=False
compute_privacy=False
product_association=False

for model in "${MODELS[@]}"; do
    echo "Evaluating model: ${model}"
    nohup python notebooks/analyze_synthetic_data.py --model ${model} --random_state ${RANDOM_STATE} --sample_size ${SAMPLE_SIZE} 
        --compute_fidelity ${compute_fidelity} &
        --compute_utility ${compute_utility} \
        --compute_privacy ${compute_privacy} \
        --product_association ${product_association}

done
