# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_regression_training_data(df):
    # define the training data for regression task
    agg_df = df.groupby(['household_id', 'basket_id', 'age', 'income', 'home_ownership',
            'marital_status', 'household_size', 'household_comp', 'kids_count', 'day']).agg({
            'quantity': 'sum',
            'unit_price': 'mean'
            }).reset_index()

    y = agg_df['quantity']
    X = agg_df.drop(columns=['quantity', 'household_id', 'basket_id', 'day'])
    return X, y

def get_classification_training_data(df, threshold=10):
    # define the training data for classification task
    agg_df = df.groupby(['household_id', 'basket_id', 'age', 'income', 'home_ownership',
            'marital_status', 'household_size', 'household_comp', 'kids_count', 'day']).agg({
            'quantity': 'sum',
            'unit_price': 'mean'
            }).reset_index()

    y = (agg_df['quantity'] > threshold).astype(int)
    X = agg_df.drop(columns=['quantity', 'household_id', 'basket_id', 'day'])
    return X, y

def train_eval_model(model, X, y, X_test, y_test, model_type='regression'):
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object']).columns

    # Define preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    # Combine preprocessing steps using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Initializing the regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', model())])

    # Training the model
    model.fit(X, y)

    # Making predictions on the testing set
    y_pred = model.predict(X_test)

    if model_type == 'regression':
        return eval_regression_model(y_test, y_pred)
    elif model_type == 'classification':
        return eval_classification_model(y_test, y_pred)

def eval_regression_model(y_test, y_pred):
    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'mse': mse, 'r2': r2}


def eval_classification_model(y_test, y_pred):
    # Evaluating the model
    accuracy = np.mean(y_test == y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    return {'accuracy': accuracy, 'f1': f1, 'roc': roc, 'precision': precision, 'recall': recall}
