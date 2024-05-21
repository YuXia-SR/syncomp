from completejourney_py import get_data
import logging
import numpy as np
import pandas as pd

class CompleteJourneyDataset():
    def __init__(self, threshold_to_remove_products=20, threshold_to_remove_large_trx=100, threshold_to_combine_products=100):
        complete_dataset = get_data()
        self.transactions = complete_dataset['transactions']
        self.transactions[['household_id', 'product_id', 'week']] = self.transactions[['household_id', 'product_id', 'week']].astype('str')
        self.products = complete_dataset['products'].astype('str')
        self.demographics = complete_dataset['demographics'].astype('str')
        self.threshold_to_remove_products = threshold_to_remove_products
        self.threshold_to_remove_large_trx = threshold_to_remove_large_trx
        self.threshold_to_combine_products = threshold_to_combine_products

    def filter_invalid_transactions(self):
        """Filter out transactions with negative quantity or negative sales amount."""
        # filter out rows with non-positive quantity sold
        self.transactions = self.transactions[
            (self.transactions["quantity"] > 0)
            & (self.transactions["sales_value"] > 0)
        ]

        logging.info(
            "Filter out transactions with non-positive quantity sold or money spent. Number of transactions are decreased to {}.".format(
                len(self.transactions)
            )
        )

        return self
    
    
    def drop_duplicate_product_id(self):
        """Assign same id to products with the same hierarchy information."""
        subset = list(self.products.columns.drop("product_id"))

        # standardize string values in datasets
        string_column = self.products.select_dtypes(include=["object"]).columns
        self.products[string_column] = self.products[
            string_column
        ].apply(lambda x: x.str.strip().str.lower().replace("\s+", " ", regex=True))

        products_no_duplicates = self.products.drop_duplicates(
            subset=subset
        ).drop(columns=["product_id"])
        products_no_duplicates.loc[:, "unique_product_id"] = np.arange(
            len(products_no_duplicates)
        )
        self.products = self.products.merge(
            products_no_duplicates, left_on=subset, right_on=subset, how="left"
        )

        self.transactions = self.transactions.merge(
            self.products[["product_id", "unique_product_id"]],
            on="product_id",
        )

        self.transactions = self.transactions.drop(
            columns=["product_id"]
        )
        self.products = self.products.drop(columns=["product_id"])
        self.transactions.rename(
            columns={"unique_product_id": "product_id"}, inplace=True
        )
        self.transactions.product_id = self.transactions.product_id.astype('str') 
        self.products = self.products.rename(
            columns={"unique_product_id": "product_id"}
        ).drop_duplicates()
        self.products.product_id = self.products.product_id.astype('str')
        self.products = self.products.fillna("None")
        self.products.sort_values("product_id", inplace=True)

        logging.info(
            "Use the same label for products with the same hierarchy information. Number of products are decreased to {}.".format(
                len(products_no_duplicates)
            )
        )
        return self
    

    def drop_duplicate_customer_id(self):
        """Assign same id to customers with the same hierarchy information."""
        subset = list(self.demographics.columns.drop("household_id"))

        # standardize string values in datasets
        string_column = self.demographics.select_dtypes(include=["object"]).columns
        self.demographics[string_column] = self.demographics[
            string_column
        ].apply(lambda x: x.str.strip().str.lower().replace("\s+", " ", regex=True))

        demographics_no_duplicates = self.demographics.drop_duplicates(
            subset=subset
        ).drop(columns=["household_id"])
        demographics_no_duplicates.loc[:, "unique_household_id"] = np.arange(
            len(demographics_no_duplicates)
        )
        self.demographics = self.demographics.merge(
            demographics_no_duplicates, left_on=subset, right_on=subset, how="left"
        )

        self.transactions = self.transactions.merge(
            self.demographics[["household_id", "unique_household_id"]],
            on="household_id",
        )

        self.transactions = self.transactions.drop(
            columns=["household_id"]
        )
        self.demographics = self.demographics.drop(columns=["household_id"])
        self.transactions.rename(
            columns={"unique_household_id": "household_id"}, inplace=True
        )
        self.transactions.household_id = self.transactions.household_id.astype('str') 
        self.demographics = self.demographics.rename(
            columns={"unique_household_id": "household_id"}
        ).drop_duplicates()
        self.demographics.household_id = self.demographics.household_id.astype('str')
        self.demographics = self.demographics.fillna("None")
        self.demographics.sort_values("household_id", inplace=True)

        logging.info(
            "Use the same label for customers with the same hierarchy information. Number of customers are decreased to {}.".format(
                len(self.demographics)
            )
        )
        return self
    
    def add_pricing_columns(self):
        """Read the raw transaction datasets.

        Add date column to prepare the prophet demand model fitting.

        Parameters
        ----------
        raw_transaction
            the base raw transaction that current object deals with

        Returns
        -------
        raw_transaction(processed)
            resulting dataframe contains two extra columns: unit price and discount portion
        """
        self.transactions = (
            self.transactions.groupby(
                ["product_id", "household_id", "week"]
            )
            .agg(
                {
                    "quantity": "sum",
                    "sales_value": "sum",
                    "retail_disc": "mean",
                    "coupon_disc": "mean",
                    "coupon_match_disc": "mean"
                }
            )
            .reset_index()
        )
        # use dealt price / quantity to get unit price
        self.transactions["unit_price"] = (
            self.transactions["sales_value"]
            / self.transactions["quantity"]
        )
        # compute shelf price, and use unit_price / shelf_price to get the discount portion
        self.transactions["base_price"] = (
            self.transactions["sales_value"]
            + self.transactions["coupon_match_disc"]
            + self.transactions["retail_disc"]
            + self.transactions["coupon_disc"]
        ) / self.transactions["quantity"]
        self.transactions["retail_discount_portion"] = (
            self.transactions["retail_disc"]
                / (self.transactions["quantity"] * self.transactions["base_price"])
        )
        self.transactions["coupon_discount_portion"] = (
            self.transactions["coupon_disc"]
                / (self.transactions["quantity"] * self.transactions["base_price"])
        )
        self.transactions["coupon_match_discount_portion"] = (
            self.transactions["coupon_match_disc"]
                / (self.transactions["quantity"] * self.transactions["base_price"])
        )
        
        return self
    
    def filter_invalid_customers(self):
        self.transactions = self.transactions[
            self.transactions["household_id"].isin(self.demographics["household_id"].unique())
        ]
        logging.info(
            "Filter out transactions with invalid customer id. Number of transactions are decreased to {}.".format(
                len(self.transactions)
            )
        )
        return self
    
    
    def filter_large_transactions(self):
        """Filter out transactions with large quantity or sales value."""
        # filter out rows with non-positive quantity sold
        self.transactions = self.transactions[
            (self.transactions["quantity"] <= self.threshold_to_remove_large_trx)
        ]

        logging.info(
            "Filter out transactions with extreme large quantity sold. Number of transactions are decreased to {}.".format(
                len(self.transactions)
            )
        )

        return self
    
    def remove_product_with_few_transactions(self):
        trx_count = self.transactions.groupby('product_id')['household_id'].count()
        products_to_remove = trx_count[trx_count < self.threshold_to_remove_products].index
        self.transactions = self.transactions[~self.transactions['product_id'].isin(products_to_remove)]
        self.products = self.products[~self.products['product_id'].isin(products_to_remove)]
        logging.info(
            f"Remove {len(products_to_remove)} products with few transactions. Number of transactions are decreased to {len(self.transactions)}."
        )
        return self


    def run_preprocess(self):
        (
            self.filter_invalid_transactions()
            .drop_duplicate_product_id()
            .drop_duplicate_customer_id()
            .add_pricing_columns()
            .filter_invalid_customers()
            .filter_large_transactions()
            .remove_product_with_few_transactions()
        )
        merged_data =  self.transactions.merge(
            self.products, on="product_id", how='inner'
        ).merge(
            self.demographics, on="household_id", how='inner'
        )

        self.products = self.products[self.products.product_id.isin(self.transactions.product_id.unique())]

        return merged_data

    
    def combine_product_with_few_transactions(self, df):
        trx_count = df.groupby('product_id')['household_id'].count().to_frame(name='weights')
        products_to_combine = trx_count[trx_count.weights < self.threshold_to_combine_products].index
        self.infrequent_products_hierarchy = self.products[self.products['product_id'].isin(products_to_combine)]
        self.infrequent_products_hierarchy = self.infrequent_products_hierarchy.merge(trx_count, left_on='product_id', right_index=True)
        self.infrequent_products_hierarchy.weights = self.infrequent_products_hierarchy.weights / self.infrequent_products_hierarchy.weights.sum()
        product_columns = self.products.columns
        combined_df = df.copy()
        combined_df.loc[combined_df['product_id'].isin(products_to_combine), product_columns] = '-1'
        
        logging.info(
            "Combine {} products with few transactions to belong to one category.".format(
                len(products_to_combine)
            )
        )
        return combined_df
    

    def load_data(self, path):
        assert path.endswith('.csv'), 'Only csv files are supported'
        data = pd.read_csv(path)
        category_columns = list(data.select_dtypes(include=['object']).columns) + ['product_id', 'household_id', 'week', 'manufacturer_id']
        data[category_columns] = data[category_columns].astype('str')

        return data


