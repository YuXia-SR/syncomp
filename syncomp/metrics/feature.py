

def compute_visit_prob_per_store(df, sample_size=1000):
    # daily number of customer visits / total number of unique customers in trx history
    daily_visits = df.groupby(['store_id', 'day']).household_id.nunique()
    total_customers = df.groupby(['store_id']).household_id.nunique()
    store_visit_prob = daily_visits / total_customers
    store_visit_prob_per_store = store_visit_prob.groupby('store_id').mean()
    sample_size = min(sample_size, len(store_visit_prob_per_store))
    return store_visit_prob_per_store.sample(sample_size)

def compute_purchase_prob_per_product(df, sample_size=1000):
    # daily number of purchase / number of unique customers in one day
    daily_purchase = df.groupby(['product_id', 'day']).household_id.nunique()
    total_customers = df.groupby(['day']).household_id.nunique()
    product_purchase_prob = daily_purchase / total_customers
    product_purchase_prob_per_product = product_purchase_prob.groupby('product_id').mean()
    sample_size = min(sample_size, len(product_purchase_prob_per_product))
    return product_purchase_prob_per_product.sample(sample_size)

def compute_basket_size_per_product(df, sample_size=1000):
    # average number of product purchased per basket
    basket_size = df.groupby('basket_id').quantity.sum()
    sample_size = min(sample_size, len(basket_size))
    return basket_size.sample(sample_size)

def compute_time_between_purchase_per_customer(df, sample_size=1000):
    # average time between purchases per customer
    visit_days = df[['household_id', 'day']].drop_duplicates().sort_values(['household_id', 'day'])
    visit_days['time_between_purchase'] = visit_days.groupby('household_id').day.diff()
    visit_days = visit_days.dropna()
    avg_time_between_purchase = visit_days.groupby('household_id').time_between_purchase.mean()
    sample_size = min(sample_size, len(avg_time_between_purchase))
    return avg_time_between_purchase.sample(sample_size)

def compute_customer_retention_per_store(df, window_length=30, sample_size=1000):
    # number of unique customers who have visited in the last 30 days / total number of unique customers
    last_day = df.day.max()
    retained_customers = df[df.day > last_day - window_length].groupby('store_id').household_id.nunique()
    customer_count = df.groupby('store_id').household_id.nunique()
    customer_retention = retained_customers / customer_count
    customer_retention = customer_retention.fillna(0)
    sample_size = min(sample_size, len(customer_retention))
    return customer_retention.sample(sample_size)

def compute_category_penetration_per_category(df, sample_size=1000):
    # number of unique customers who have purchased one category / total number of unique customers
    customer_counts = df.groupby('product_category').household_id.nunique()
    total_customers = df.household_id.nunique()
    category_penetration = customer_counts / total_customers
    sample_size = min(sample_size, len(category_penetration))
    return category_penetration.sample(sample_size)

def get_column(df, column_name, sample_size=1000):
    # get a sample of a column from a dataframe
    if df[column_name].dtype == 'object':
        return df[column_name].fillna(df[column_name].mode()[0]).sample(sample_size)
    else:
        return df[column_name].fillna(df[column_name].mean()).sample(sample_size)

def compute_product_price(df):
    # compute unit price and base price
    df['unit_price'] = df['sales_value'] / df['quantity']
    df['base_price'] = (df['sales_value'] + df['coupon_match_disc'] + df['retail_disc']) / df['quantity']
    return df