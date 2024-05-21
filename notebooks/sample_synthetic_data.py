import pandas as pd
import logging
import argparse
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from syncomp.utils.data_util import CompleteJourneyDataset
from syncomp.utils.holdout_util import split_dataframe
from syncomp.utils.train_util import train_autodiff, train_ctgan, train_ctabgan

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('rdt').setLevel(logging.WARNING)

def format_infrequent_product_in_synthetic_data(df, infrequent_products_hierarchy, weights=None):
    product_hierarchy = infrequent_products_hierarchy.sample(len(df), replace=True, weights=weights)
    product_columns = product_hierarchy.columns
    df.loc[:, product_columns] = product_hierarchy

    return df.drop(columns=['product_id'])

def train_func(department, train_df, train, infrequent_products_hierarchy, infrequent_products_weights):
    filtered_train_df = train_df[train_df.department == department]

    unique_cols = [col for col in filtered_train_df.columns if filtered_train_df[col].nunique() == 1]
    dropped_cols = filtered_train_df[unique_cols]
    filtered_train_df_reduced = filtered_train_df.drop(columns=unique_cols)
    logging.info(f"Training dataset for department {department}: {filtered_train_df_reduced.shape}")
    filtered_syn_df_reduced = train(filtered_train_df_reduced)
    filtered_syn_df = pd.concat([filtered_syn_df_reduced, dropped_cols], axis=-1)
    if department == '-1':
        filtered_syn_df = format_infrequent_product_in_synthetic_data(filtered_syn_df, infrequent_products_hierarchy, infrequent_products_weights)

    return filtered_syn_df

def main(
    model: str='AutoDiff',
    random_state: int=0,
    df_split_ratio: list=[0.4, 0.4, 0.2],
    dir:str ='results',
    n_job: int=4,
):
    logging.info('args: model=%s, random_state=%s, df_split_ratio=%s', model, random_state, df_split_ratio)
    logging.info('read real df')
    cd = CompleteJourneyDataset()
    real_df = cd.run_preprocess()
    real_df_combined = cd.combine_product_with_few_transactions(real_df)
    train_df, _, _ = split_dataframe(real_df_combined, df_split_ratio, random_state, groupby='department')
    infrequent_products_hierarchy = cd.infrequent_products_hierarchy
    infrequent_products_weights = infrequent_products_hierarchy['weights']
    infrequent_products_hierarchy = infrequent_products_hierarchy.drop(columns=['weights'])

    logging.info(f'generate one syn_df using {model}')
    if model == 'AutoDiff':
        train = train_autodiff
    elif model == 'CTGAN':
        train = train_ctgan
    elif model == 'CTABGAN':
        train = train_ctabgan

    department_group = train_df['department'].unique()
    syn_df = Parallel(n_jobs=n_job)(delayed(train_func)(
        department, train_df, train, 
        infrequent_products_hierarchy, infrequent_products_weights
    ) for department in department_group)

    syn_df = pd.concat(syn_df)

    categorical_columns = real_df.select_dtypes(include=['object']).columns
    syn_df[categorical_columns] = syn_df[categorical_columns].astype(str)
    int_columns = real_df.select_dtypes(include=['int64']).columns
    syn_df[int_columns] = syn_df[int_columns].astype(int)

    logging.info('save synthetic data')
    os.makedirs(f'{dir}/{model}/{random_state}', exist_ok=True)
    syn_df.to_csv(f'{dir}/{model}/{random_state}/synthetic_data.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A simple program with argparse")
    parser.add_argument('--model', type=str, help='Model to use for generating synthetic data', default='AutoDiff')
    parser.add_argument('--random_state', type=int, help='Random state to split the real data', default=0)
    parser.add_argument('--df_split_ratio', type=float, nargs='+', help='Proportions to split the real data', default=[0.4, 0.4, 0.2])
    parser.add_argument('--dir', type=str, help='Directory to save the result', default='results')
    parser.add_argument('--customer_batch_size', type=int, help='Number of unique customers in one training group', default=500)
    parser.add_argument('--n_job', type=int, help='Number of simulation running in parallel', default=4)
    
    args = parser.parse_args()

    main(args.model, args.random_state, args.df_split_ratio, args.dir)