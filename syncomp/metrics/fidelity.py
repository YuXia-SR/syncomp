from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from scipy.spatial.distance import euclidean
import numpy as np

from syncomp.metrics.feature import (
    compute_basket_size_per_product,
    compute_category_penetration_per_category,
    compute_customer_retention,
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
    demographics_columns = ['age', 'income', 'home_ownership', 'marital_status', 'household_size', 'household_comp', 'kids_count']
    demographics = df[demographics_columns].drop_duplicates().reset_index(drop=True)
    demographics['household_id'] = range(len(demographics))
    hierarchy_columns = ['manufacturer_id', 'department', 'brand', 'product_category', 'product_type', 'package_size']
    hierarchy = df[hierarchy_columns].drop_duplicates().reset_index(drop=True)
    hierarchy['product_id'] = range(len(hierarchy))
    merged_df = df.merge(demographics, on=demographics_columns, how='left'). \
        merge(hierarchy, on=hierarchy_columns, how='left')
    # Extract feature distribution
    numerical_feature = {
        "visit_prob_per_store": compute_visit_prob_per_store(merged_df, sample_size=sample_size),
        "purchase_prob_per_product": compute_purchase_prob_per_product(merged_df, sample_size=sample_size),
        "basket_size_per_product": compute_basket_size_per_product(merged_df, sample_size=sample_size),
        "time_between_purchase_per_customer": compute_time_between_purchase_per_customer(merged_df, sample_size=sample_size),
        "customer_retention_per_store": compute_customer_retention(merged_df, sample_size=sample_size),
        "category_penetration_per_category": compute_category_penetration_per_category(merged_df, sample_size=sample_size),
        "retail_disc": get_column(merged_df, "retail_disc", sample_size=sample_size),
        "coupon_disc": get_column(merged_df, "coupon_disc", sample_size=sample_size),
        "coupon_match_disc": get_column(merged_df, "coupon_match_disc", sample_size=sample_size),
        "revenue": get_column(merged_df, "sales_value", sample_size=sample_size),
        "quantity": get_column(merged_df, "quantity", sample_size=sample_size)
    }
    categorical_feature = {
        "age": get_column(merged_df, "age", sample_size=sample_size),
        "income": get_column(merged_df, "income", sample_size=sample_size),
        "household_size": get_column(merged_df, "household_size", sample_size=sample_size),
        "kids_count": get_column(merged_df, "kids_count", sample_size=sample_size),
        "product_category": get_column(merged_df, "product_category", sample_size=sample_size),
        "department": get_column(merged_df, "department", sample_size=sample_size),
    }
    # Compute corr, theils_u, and ratio matrices
    continuous_columns, categorical_columns = detect_column_types(merged_df)
    pearson_coef, theils_u, correl_ratio = compute_correlation(merged_df, continuous_columns, categorical_columns, sample_size=sample_size, exclude_columns=exclude_columns)
    corr_feature = {
        "pearson_coef": pearson_coef,
        "theils_u": theils_u,
        "correl_ratio": correl_ratio
    }

    return {'numerical': numerical_feature, 'categorical': categorical_feature, 'interaction': corr_feature}

def string_conv_int(x):
    mapping = {v: i for i, v in enumerate(set(x))}
    return np.array(list(map(mapping.__getitem__, x)))

def compare_fidelity_info(real_fidelity_info, syn_fidelity_info):
    real_numerical_feature, real_categorical_feature, real_corr_feature = real_fidelity_info['numerical'], real_fidelity_info['categorical'], real_fidelity_info['interaction']
    syn_numerical_feature, syn_categorical_feature, syn_corr_feature = syn_fidelity_info['numerical'], syn_fidelity_info['categorical'], syn_fidelity_info['interaction']
    
    wasserstein_distance_metric = {}
    jensenshannon_distance_metric = {}
    euclidean_distance_metric = {}

    for feature_name in real_numerical_feature.keys():
        wasserstein_distance_metric[feature_name] = wasserstein_distance(real_numerical_feature[feature_name], syn_numerical_feature[feature_name])
    
    for feature_name in real_categorical_feature.keys():
        real_feature = string_conv_int(real_categorical_feature[feature_name])
        syn_feature = string_conv_int(syn_categorical_feature[feature_name])
        jensenshannon_distance_metric[feature_name] = jensenshannon(real_feature, syn_feature)


    for corr_name in real_corr_feature.keys():
        euclidean_distance_metric[corr_name] = euclidean(real_corr_feature[corr_name].fillna(0).values.flatten(), syn_corr_feature[corr_name].fillna(0).values.flatten())

    return {
        'wasserstein_distance': wasserstein_distance_metric,
        'jensenshannon_distance': jensenshannon_distance_metric,
        'euclidean_distance': euclidean_distance_metric
    }