import pandas as pd

def split_dataframe(df, proportions, random_state=0):
    # Calculate the number of rows for each split
    total_rows = len(df)
    split_sizes = [int(total_rows * p) for p in proportions]
    
    # Ensure the sum of proportions is 1
    assert sum(proportions) == 1.0, "Proportions should sum to 1.0"
    
    # Shuffle the DataFrame
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Split the DataFrame
    splits = []
    start = 0
    for size in split_sizes:
        splits.append(df_shuffled.iloc[start:start+size].reset_index(drop=True))
        start += size
    
    return splits
