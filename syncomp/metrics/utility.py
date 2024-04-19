# Importing necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def get_regression_training_data(df):
    # define the training data for regression task
    agg_df = df.groupby(['household_id', 'product_id', 'age', 'income', 'home_ownership',
            'marital_status', 'household_size', 'household_comp', 'kids_count', 'day']).agg({
            'quantity': 'sum',
            'unit_price': 'mean'
            }).reset_index()

    y = agg_df['quantity']
    X = agg_df.drop(columns=['quantity'])
    return X, y

def train_eval_regression_model(model, X, y):
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

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing the regression model
    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', model())])

    # Training the model
    model.fit(X_train, y_train)

    # Making predictions on the testing set
    y_pred = model.predict(X_test)

    # Evaluating the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {'mse': mse, 'r2': r2}

