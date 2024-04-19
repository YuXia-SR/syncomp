from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
import numpy as np

from syncomp.metrics.feature import (
    compute_basket_size_per_product,
    compute_category_penetration_per_category,
    compute_customer_retention_per_store,
    compute_product_price,
    compute_purchase_prob_per_product,
    compute_time_between_purchase_per_customer,
    compute_visit_prob_per_store,
    get_column,
)
from syncomp.metrics.corr import (
    detect_column_types,
    compute_correlation
)

def gather_fidelity_info_from_df(df, sample_size=1000, exclude_columns=[]):
    # Get product price
    df = compute_product_price(df)
    # Extract feature distribution
    numerical_feature = {
        "visit_prob_per_store": compute_visit_prob_per_store(df, sample_size=sample_size),
        "purchase_prob_per_product": compute_purchase_prob_per_product(df, sample_size=sample_size),
        "basket_size_per_product": compute_basket_size_per_product(df, sample_size=sample_size),
        "time_between_purchase_per_customer": compute_time_between_purchase_per_customer(df, sample_size=sample_size),
        "customer_retention_per_store": compute_customer_retention_per_store(df, sample_size=sample_size),
        "category_penetration_per_category": compute_category_penetration_per_category(df, sample_size=sample_size),
        "unit_price": get_column(df, "unit_price", sample_size=sample_size),
        "base_price": get_column(df, "base_price", sample_size=sample_size),
        "revenue": get_column(df, "sales_value", sample_size=sample_size),
        "quantity": get_column(df, "quantity", sample_size=sample_size)
    }
    categorical_feature = {
        "age": get_column(df, "age", sample_size=sample_size),
        "income": get_column(df, "income", sample_size=sample_size),
        "household_size": get_column(df, "household_size", sample_size=sample_size),
        "kids_count": get_column(df, "kids_count", sample_size=sample_size),
        "product_category": get_column(df, "product_category", sample_size=sample_size),
        "department": get_column(df, "department", sample_size=sample_size),
    }
    # Compute corr, theils_u, and ratio matrices
    continuous_columns, categorical_columns = detect_column_types(df)
    pearson_coef, theils_u, correl_ratio = compute_correlation(df, continuous_columns, categorical_columns, sample_size=sample_size, exclude_columns=exclude_columns)
    corr_feature = {
        "pearson_coef": pearson_coef,
        "theils_u": theils_u,
        "correl_ratio": correl_ratio
    }

    return numerical_feature, categorical_feature, corr_feature

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

def compare_fidelity_info(real_fidelity_info, syn_fidelity_info):
    real_numerical_feature, real_categorical_feature, real_corr_feature = real_fidelity_info
    syn_numerical_feature, syn_categorical_feature, syn_corr_feature = syn_fidelity_info
    
    wasserstein_distance_metric = {}
    jensenshannon_distance_metric = {}

    for feature_name in real_numerical_feature.keys():
        wasserstein_distance_metric[feature_name] = wasserstein_distance(real_numerical_feature[feature_name], syn_numerical_feature[feature_name])
    
    for feature_name in real_categorical_feature.keys():
        real_feature = string_conv_int(real_categorical_feature[feature_name])
        syn_feature = string_conv_int(syn_categorical_feature[feature_name])
        jensenshannon_distance_metric[feature_name] = jensenshannon(real_feature, syn_feature)

    return wasserstein_distance_metric, jensenshannon_distance_metric, real_corr_feature, syn_corr_feature