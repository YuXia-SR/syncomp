import pandas as pd
import logging
import argparse
import os

from syncomp.utils.holdout_util import split_dataframe
from syncomp.utils.train_util import train_autodiff, train_ctgan, train_ctabgan

logging.getLogger().setLevel(logging.INFO)
logging.getLogger('rdt').setLevel(logging.WARNING)

def main(
    model: str='AutoDiff',
    random_state: int=0,
    df_split_ratio: list=[0.4, 0.4, 0.2],
    dir ='results'
):
    logging.info('args: model=%s, random_state=%s, df_split_ratio=%s', model, random_state, df_split_ratio)
    logging.info('read real df')
    real_df = pd.read_csv(f'{dir}/complete_dataset_filtered.csv', converters={'household_size': str, 'kids_count': str})
    train_df, _, _ = split_dataframe(real_df, df_split_ratio, random_state)

    logging.info(f'generate one syn_df using {model}')
    if model == 'AutoDiff':
        syn_df, _ = train_autodiff(train_df=train_df)
    elif model == 'CTGAN':
        syn_df = train_ctgan(train_df=train_df, epochs=10)
    elif model == 'CTABGAN':
        syn_df = train_ctabgan(train_df=train_df)
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