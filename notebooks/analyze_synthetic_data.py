import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import BaggingClassifier
import logging
import argparse
import pickle
import json
import os

from syncomp.metrics.fidelity import (
    gather_fidelity_info_from_df,
    compare_fidelity_info
)
from syncomp.metrics.utility import (
    get_regression_training_data,
    get_classification_training_data,
    train_eval_model
)
from syncomp.metrics.privacy import distance_closest_record, distance_closest_record_comparison
from syncomp.utils.holdout_util import split_dataframe
from syncomp.utils.data_util import CompleteJourneyDataset

def set_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger('rdt').setLevel(logging.WARNING)

def main(    
    model: str='AutoDiff',
    random_state: int=0,
    df_split_ratio: list=[0.4, 0.4, 0.2],
    sample_size = 1000,
    dir ='results',
    compute_fidelity=True,
    compute_utility=True,
    compute_privacy=True
):
    set_logging()

    # read real df and synthetic df
    cd = CompleteJourneyDataset()
    real_df = cd.run_preprocess()
    real_df = real_df.drop(columns=['product_id', 'household_id'])
    train_df, holdout_df, eval_df = split_dataframe(real_df, df_split_ratio, random_state)

    # generate one syn_df using autodiff
    syn_files = os.listdir(f'{dir}/{model}/{random_state}')
    syn_df = []
    for file in syn_files:
        if file.endswith('.csv') and 'privacy_distance_comparison.csv' not in file:
            df = cd.load_data(f'{dir}/{model}/{random_state}/{file}')
            syn_df.append(df)
    syn_df = pd.concat(syn_df)

    """ 
    Fidelity metrics
    """
    if not compute_fidelity:
        logging.info('Skip computing fidelity metrics')
    else:
        logging.info('Compute fidelity metrics')
        train_fidelity_info = gather_fidelity_info_from_df(train_df, sample_size=sample_size, exclude_columns=['household_id', 'basket_id'])
        syn_fidelity_info = gather_fidelity_info_from_df(syn_df, sample_size=sample_size, exclude_columns=['household_id', 'basket_id'])
        holdout_fidelity_info = gather_fidelity_info_from_df(holdout_df, sample_size=sample_size, exclude_columns=['household_id', 'basket_id'])

        fidelity_metrics_syn = compare_fidelity_info(train_fidelity_info, syn_fidelity_info)
        fidelity_metrics_holdout = compare_fidelity_info(train_fidelity_info, holdout_fidelity_info)

        fidelity_info = {
            'train_df': train_fidelity_info,
            'syn_df': syn_fidelity_info,
            'holdout_df': holdout_fidelity_info
        }
        fidelity_metric = {'syn_df': fidelity_metrics_syn, 'holdout_df': fidelity_metrics_holdout}
        with open(f'{dir}/{model}/{random_state}/fidelity_info.pkl', 'wb') as f:
            pickle.dump(fidelity_info, f)
        with open(f'{dir}/{model}/{random_state}/fidelity_metrics.json', 'w') as f:
            json.dump(fidelity_metric, f, indent=4)

    """
    Utility metrics
    """
    if not compute_utility:
        logging.info('Skip computing utility metrics')
    else:
        logging.info('Compute utility metrics')
        classification_model = BaggingClassifier
        regression_model = PoissonRegressor

        # classification task
        train_X, train_y = get_classification_training_data(train_df)
        syn_X, syn_y = get_classification_training_data(syn_df)
        holdout_X, holdout_y = get_classification_training_data(holdout_df)
        eval_X, eval_y = get_classification_training_data(eval_df)
        classification_real_metric = train_eval_model(classification_model, train_X, train_y, eval_X, eval_y, model_type='classification')
        classification_holdout_metric = train_eval_model(classification_model, holdout_X, holdout_y, eval_X, eval_y, model_type='classification')
        classification_syn_metric = train_eval_model(classification_model, syn_X, syn_y, eval_X, eval_y, model_type='classification')

        # regression task
        train_X, train_y = get_regression_training_data(train_df)
        syn_X, syn_y = get_regression_training_data(syn_df)
        holdout_X, holdout_y = get_regression_training_data(holdout_df)
        eval_X, eval_y = get_regression_training_data(eval_df)
        # train a linear regression model
        regression_real_metric = train_eval_model(regression_model, train_X, train_y, eval_X, eval_y)
        regression_holdout_metric = train_eval_model(regression_model, holdout_X, holdout_y, eval_X, eval_y)
        regression_syn_metric = train_eval_model(regression_model, syn_X, syn_y, eval_X, eval_y)

        utility_metrics = {
            'classification': {'train_df': classification_real_metric, 'holdout_df': classification_holdout_metric, 'syn_df': classification_syn_metric},
            'regression': {'train_df': regression_real_metric, 'holdout_df': regression_holdout_metric, 'syn_df': regression_syn_metric}
        }
        with open(f'{dir}/{model}/{random_state}/utility_metrics.json', 'w') as f:
            json.dump(utility_metrics, f, indent=4)

    """
    Privacy metrics
    """
    if not compute_privacy:
        logging.info('Skip computing privacy metrics')
    else:
        logging.info('Compute privacy metrics')
        dcr = distance_closest_record(train_df, syn_df)
        # tapas_attack = evaluate_tapas_attack(train_df, model, random_state, dir=dir, n_sample=1000, num_training_records=10)
        distance_comparison = distance_closest_record_comparison(train_df, syn_df, holdout_df, sample_size=sample_size, random_state=random_state)

        distance_comparison.to_csv(f'{dir}/{model}/{random_state}/privacy_distance_comparison.csv', index=False)
        # with open(f'{dir}/{model}/{random_state}/privacy_metrics.pkl', 'wb') as f:
        #     pickle.dump({'dcr': dcr, 'tapas': tapas_attack}, f)
        with open(f'{dir}/{model}/{random_state}/privacy_metrics.json', 'w') as f:
            json.dump({'dcr': dcr}, f, indent=4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple program with argparse")
    parser.add_argument('--model', type=str, help='Model to use for generating synthetic data', default='TabAutoDiff')
    parser.add_argument('--random_state', type=int, help='Random state to split the real data', default=0)
    parser.add_argument('--df_split_ratio', type=float, nargs='+', help='Proportions to split the real data', default=[0.4, 0.4, 0.2])
    parser.add_argument('--sample_size', type=int, help='Sample size for fidelity and privacy metrics', default=2000)
    parser.add_argument('--dir', type=str, help='Directory to save the result', default='results')
    parser.add_argument('--compute_fidelity', type=bool, help='Compute fidelity metrics', default=True)
    parser.add_argument('--compute_utility', type=bool, help='Compute utility metrics', default=True)
    parser.add_argument('--compute_privacy', type=bool, help='Compute privacy metrics', default=True)
    args = parser.parse_args()

    main(
        args.model, args.random_state, args.df_split_ratio, 
        args.sample_size, args.dir, args.compute_fidelity, 
        args.compute_utility, args.compute_privacy
    )