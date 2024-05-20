import pandas as pd
import logging
import argparse
import os
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

    return df

def main(
    model: str='AutoDiff',
    random_state: int=0,
    df_split_ratio: list=[0.4, 0.4, 0.2],
    dir ='results',
    household_id_unique_threshold=500
):
    logging.info('args: model=%s, random_state=%s, df_split_ratio=%s', model, random_state, df_split_ratio)
    logging.info('read real df')
    cd = CompleteJourneyDataset()
    real_df = cd.run_preprocess()
    real_df_combined = cd.combine_product_with_few_transactions(real_df)
    train_df, _, _ = split_dataframe(real_df_combined, df_split_ratio, random_state)
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

    product_category_group = train_df['product_category'].unique()
    for category in tqdm(product_category_group, desc='Generate synthetic data for each category'):
        filtered_train_df = train_df[train_df.product_category == category]

        if filtered_train_df.household_id.nunique() > household_id_unique_threshold:
            unique_household_id = filtered_train_df.household_id.unique()
            n_household_id = len(unique_household_id)

            start_idx = 0
            end_idx = household_id_unique_threshold
            while end_idx <= n_household_id:
                sample_household_id = unique_household_id[start_idx: end_idx]
                sampled_filtered_train_df = train(filtered_train_df[filtered_train_df.household_id.isin(sample_household_id)])
                filtered_syn_df = train(sampled_filtered_train_df)
                if category == '-1':
                    filtered_syn_df = format_infrequent_product_in_synthetic_data(filtered_syn_df, infrequent_products_hierarchy, infrequent_products_weights)
                syn_df.append(filtered_syn_df)
                start_idx = end_idx
                end_idx += household_id_unique_threshold
        else:
            filtered_syn_df = train(filtered_train_df)
            if category == '-1':
                filtered_syn_df = format_infrequent_product_in_synthetic_data(filtered_syn_df, infrequent_products_hierarchy, infrequent_products_weights)
            syn_df.append(filtered_syn_df)

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
    args = parser.parse_args()

    main(args.model, args.random_state, args.df_split_ratio, args.dir)