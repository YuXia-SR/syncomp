
MODELS=("TabDDPM" "TabAutoDiff" "StasyAutoDiff" "CTGAN" "CTABGAN" "Stasy" "AutoGAN")
RANDOM_STATE=0
SAMPLE_SIZE=10000

for model in "${MODELS[@]}"; do
    echo "Evaluating model: ${model}"
    python notebooks/analyze_synthetic_data.py --model ${model} --random_state ${RANDOM_STATE} --sample_size ${SAMPLE_SIZE}
done
