# Synthesizer Evaluation on Retail Dataset

This repository is the codebase for paper "Advancing Retail Data Science: Comprehensive Evaluation of Synthetic Data" accepted to GenAI Evaluation KDD2024.

## Abstract
The evaluation of synthetic data generation is crucial, especially in the retail sector where data accuracy is paramount. This paper introduces a comprehensive framework for assessing synthetic retail data, focusing on fidelity, utility, and privacy. Our approach differentiates between continuous and discrete data attributes, providing precise evaluation criteria. Fidelity is measured through stability and generalizability. Stability ensures synthetic data accurately replicates known data distributions, while generalizability confirms its robustness in novel scenarios. Utility is demonstrated through the synthetic data's effectiveness in critical retail tasks such as demand forecasting and dynamic pricing, proving its value in predictive analytics and strategic planning. Privacy is safeguarded using Differential Privacy, ensuring synthetic data maintains a perfect balance between resembling training and holdout datasets without compromising security. Our findings validate that this framework provides reliable and scalable evaluation for synthetic retail data. It ensures high fidelity, utility, and privacy, making it an essential tool for advancing retail data science. This framework meets the evolving needs of the retail industry with precision and confidence, paving the way for future advancements in synthetic data methodologies.

## Experiment
To replicate the experiment, please follow the steps below:

1. Install the required packages using poetry:
```bash
poetry install
poetry shell
```

2. Run the experiment and the experiment results will be saved in the `results` folder:
```bash
sh train.sh
sh eval.sh
```