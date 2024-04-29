from completejourney_py import get_data
import pandas as pd
import logging

def load_data(mode='download', **kwargs):
    if mode == 'download':
        logging.info('Downloading data from python api')
        complete_dataset = get_data()
        merged_data = complete_dataset["transactions"].merge(
            complete_dataset["products"], how="inner"
        ).merge(
            complete_dataset["demographics"], how="inner"
        ).merge(
            complete_dataset["campaigns"], how="left"
        ).merge(
            complete_dataset["coupons"], how="left"
        ).merge(
            complete_dataset["coupon_redemptions"], how="left"
        ).merge(
            complete_dataset["campaign_descriptions"], how="left"
        ).merge(
            complete_dataset["promotions"], how="left"
        )
        return merged_data
    elif mode == 'local':
        logging.info('Loading data from local file')
        path = kwargs.get('path') or ValueError('Path not provided')
        return pd.read_parquet(path)
    else:
        raise ValueError(f'Unknown mode: {mode}')
    
def filter_data(df, max_quantity=200, min_visit_days=24, min_unique_customers=100, min_purchase_count=100):

    logging.info('Cleaning data columns')
    start = df['transaction_timestamp'].min()
    df['day'] = (df['transaction_timestamp'] - start).dt.days + 1
    df['redemption_day'] = (df['redemption_date'] - start).dt.days + 1
    timestamp_columns = df.select_dtypes(include=['datetime64']).columns
    df = df.drop(columns=timestamp_columns)
    df = df.drop(columns=['coupon_upc'])
    id_columns = [column for column in df.columns if '_id' in column]
    df[id_columns] = df[id_columns].fillna(0).astype('int')

    logging.info(f"Filtering out unvalid transaction")
    df = df[df.quantity > 0]

    logging.info(f"Filtering out trx with extremely large quantity")
    df = df[df.quantity <= max_quantity]

    logging.info(f"Filtering out customers visit store less than {min_visit_days} days")
    visit_days = df.groupby('household_id')['day'].nunique()
    selected_customer = visit_days[visit_days >= min_visit_days].index
    df = df[df['household_id'].isin(selected_customer)]
    logging.info(f"Number of trx: {len(df)}")

    logging.info(f"Filter out products that have been purchased by less than {min_unique_customers} customers")
    unique_customers = df.groupby('product_id')['household_id'].nunique()
    selected_product = unique_customers[unique_customers >= min_unique_customers].index
    df = df[df['product_id'].isin(selected_product)]
    logging.info(f"Number of trx: {len(df)}")

    logging.info(f"Filter out products that have been purchased less than {min_purchase_count} times")
    purchase_count = df.groupby('product_id')['day'].count()
    selected_product = purchase_count[purchase_count >= min_purchase_count].index
    df = df[df['product_id'].isin(selected_product)]
    logging.info(f"Number of trx: {len(df)}")
    
    logging.info(f"Fill missing values for the redemption_day column")
    df['redemption_day'] = df['redemption_day'].fillna(0).astype('int')

    logging.info(f"Fill missing values for string columns")
    string_columns = df.select_dtypes(include=['object']).columns
    df[string_columns] = df[string_columns].astype(str).fillna('BLANK')
    
    return df.reset_index(drop=True)

def compute_unit_price(df):
    df['unit_price'] = df['sales_value'] / df['quantity']
    return df

