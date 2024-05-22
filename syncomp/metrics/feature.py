import numpy as np

def compute_visit_prob_per_store(df, sample_size=1000, random_state=0):
    # daily number of customer visits / total number of unique customers in trx history
    daily_visits = df.groupby(['week']).household_id.nunique()
    total_customers = df.household_id.nunique()
    store_visit_prob = daily_visits / total_customers
    sample_size = min(sample_size, len(store_visit_prob))
    return store_visit_prob.sample(sample_size, random_state=random_state)

def compute_purchase_prob_per_product(df, sample_size=1000, random_state=0):
    # daily number of purchase / number of unique customers in one day
    daily_purchase = df.groupby(['product_id']).household_id.nunique()
    total_customers = df.household_id.nunique()
    product_purchase_prob = daily_purchase / total_customers
    sample_size = min(sample_size, len(product_purchase_prob))
    return product_purchase_prob.sample(sample_size, random_state=random_state)

def compute_basket_size_per_product(df, sample_size=1000, random_state=0):
    # average number of product purchased per basket
    basket_size = df.groupby(['week', 'household_id']).quantity.sum()
    sample_size = min(sample_size, len(basket_size))
    return basket_size.sample(sample_size, random_state=random_state)

def compute_time_between_purchase_per_customer(df, sample_size=1000, random_state=0):
    # average time between purchases per customer
    visit_days = df[['household_id', 'week']].drop_duplicates().sort_values(['household_id', 'week'])
    visit_days['week'] = visit_days['week'].astype('float')
    visit_days['time_between_purchase'] = visit_days.groupby('household_id').week.diff()
    visit_days = visit_days.dropna()
    avg_time_between_purchase = visit_days.groupby('household_id').time_between_purchase.mean()
    sample_size = min(sample_size, len(avg_time_between_purchase))
    return avg_time_between_purchase.sample(sample_size, random_state=random_state)

def compute_customer_retention(df, window_length=30, sample_size=1000, random_state=0):
    # number of unique customers who have visited in the last 30 days / total number of unique customers
    last_day = max(df.week.astype('float'))
    retained_customers = df[df.week.astype('float') > last_day - window_length].household_id.nunique()
    customer_count = df.household_id.nunique()
    customer_retention = retained_customers / customer_count
    return np.asarray([customer_retention])

def compute_category_penetration_per_category(df, sample_size=1000, random_state=0):
    # number of unique customers who have purchased one category / total number of unique customers
    customer_counts = df.groupby('product_category').household_id.nunique()
    total_customers = df.household_id.nunique()
    category_penetration = customer_counts / total_customers
    sample_size = min(sample_size, len(category_penetration))
    return category_penetration.sample(sample_size, random_state=random_state)

def get_column(df, column_name, sample_size=1000, random_state=0):
    # get a sample of a column from a dataframe
    if df[column_name].dtype == 'object':
        return df[column_name].fillna(df[column_name].mode()[0]).sample(sample_size, random_state=random_state)
    else:
        return df[column_name].fillna(df[column_name].mean()).sample(sample_size, random_state=random_state)