# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 08:59:16 2023
@author: Namjoon Suh
"""
import pandas as pd
import numpy as np
from dython.nominal import theils_u

def detect_column_types(dataframe):
    continuous_columns = []
    categorical_columns = []

    for col in dataframe.columns:
        # Calculate the ratio of unique values to the total number of rows
        n_unique = dataframe[col].nunique()

        # If the ratio is below the threshold, consider the column as categorical
        if n_unique >= 2 and n_unique <= 25 and dataframe[col].dtype == 'int64':
            categorical_columns.append(col)
        elif dataframe[col].dtype == 'object':
            categorical_columns.append(col)
        else:
            continuous_columns.append(col)

    return continuous_columns, categorical_columns

def theils_u_mat(df):
  # Compute Theil's U-statistics between each pair of columns
  cate_columns = df.shape[1]
  theils_u_mat = np.zeros((cate_columns, cate_columns))

  for i in range(cate_columns):
      for j in range(cate_columns):
          theils_u_mat[i, j] = theils_u(df.iloc[:, i], df.iloc[:, j])

  return theils_u_mat

# See the post https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def correlation_ratio(categories, measurements):
    fcat, _ = pd.factorize(categories)
    cat_num = np.max(fcat)+1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0,cat_num):
        cat_measures = measurements[np.argwhere(fcat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array,n_array))/np.sum(n_array)
    numerator = np.sum(np.multiply(n_array,np.power(np.subtract(y_avg_array,y_total_avg),2)))
    denominator = np.sum(np.power(np.subtract(measurements,y_total_avg),2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = numerator/denominator
    return eta

def ratio_mat(df, continuous_columns, categorical_columns):    
    if len(categorical_columns) == 0 or len(continuous_columns) == 0:
        return np.zeros(1)
    else:
        rat_mat = np.ones((len(continuous_columns), len(categorical_columns)))
        for i, cat_col in enumerate(categorical_columns):
            for j, cont_col in enumerate(continuous_columns):
                rat_mat[j][i] = correlation_ratio(df[cat_col], df[cont_col])
        return rat_mat

def fillNa_cont(df):
    for col in df.columns:
        mean_values = df[col].mean()
        df[col].fillna(mean_values, inplace=True)
    return df

def fillNa_cate(df):
    for col in df.columns:
        mode_values = df[col].mode()[0]
        df[col].fillna(mode_values, inplace=True)
    return df

def compute_correlation(df, continuous_columns, categorical_columns, sample_size=1000, exclude_columns=[]):

    num_mat = pd.DataFrame(df[continuous_columns])
    cat_mat = pd.DataFrame(df[categorical_columns])
    
    sample_size = min(sample_size, len(num_mat))
    num_mat = fillNa_cont(num_mat).sample(sample_size)
    cat_mat = fillNa_cate(cat_mat).sample(sample_size)
    
    pearson_sub_matrix = np.corrcoef(num_mat, rowvar = False)
    pearson_sub_matrix = pd.DataFrame(pearson_sub_matrix, columns=continuous_columns, index=continuous_columns)
    
    theils_u_matrix = theils_u_mat(cat_mat)
    theils_u_matrix = pd.DataFrame(theils_u_matrix, columns=categorical_columns, index=categorical_columns)
    
    for col in exclude_columns:
        if col in continuous_columns:
            continuous_columns.remove(col)
        elif col in categorical_columns:
            categorical_columns.remove(col)
    correl_ratio_mat = ratio_mat(df, continuous_columns, categorical_columns)
    correl_ratio_mat = pd.DataFrame(correl_ratio_mat, columns=categorical_columns, index=continuous_columns)
   
    return (pearson_sub_matrix, theils_u_matrix, correl_ratio_mat)