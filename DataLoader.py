#imports
import pandas as pd
import numpy as np


#load data
data_fraud = pd.read_csv(r'data/fraud.csv')

# Removing rows containing missing values
data_fraud = data_fraud.dropna(how='any')


def corrupt_labels(df, target_col, corruption_rate, random_state=1):
    """
    Adds noise to the dataframe (df) by corrupting a certain percentage (corruption_rate)
    of the values in the target column (target_col).


    :param df: pandas DataFrame
    :param target_col: name of the target column to corrupt
    :param corruption_rate: fraction of labels to corrupt (0 to 1)
    :param random_state: random state for reproducibility
    """
    rng = np.random.default_rng(random_state)   # For reproducibility
    df_new = df.copy()
    n = len(df_new)


    idx = rng.choice(df_new.index, size=int(corruption_rate * n), replace=False)

    unique_classes = df_new[target_col].unique() # possible classes from the target column; 0 or 1

    for i in idx:
        current = df_new.at[i, target_col]
        possible = [c for c in unique_classes if c != current]

        if not possible:
            continue    # nothing to change (single-class column)
        df_new.at[i, target_col] = rng.choice(possible)

    return df_new

