def split_dataframe(df, proportions=[0.4, 0.4, 0.2], random_state=0, groupby=None):

    splits = []
    current_df = df.copy()
    for i, proportion in enumerate(proportions):
        frac = proportion / sum(proportions[i:])
        if groupby is None:
            sample_df = current_df.sample(frac=frac, random_state=random_state)
        else:
            sample_df = current_df.groupby(groupby).sample(frac=frac, random_state=random_state)
        current_df = current_df.drop(sample_df.index)
        splits.append(sample_df.reset_index(drop=True))
    
    return splits
