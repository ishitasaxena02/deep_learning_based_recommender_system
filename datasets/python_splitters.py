import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split as sk_split
from recommenders.utils.constants import (  DEFAULT_ITEM_COL, DEFAULT_USER_COL, DEFAULT_TIMESTAMP_COL,)
from recommenders.datasets.split_utils import ( process_split_ratio,  min_rating_filter_pandas, split_pandas_data_with_ratios,)

def numpy_stratified_split(X, ratio=0.75, seed=42):
    #This splits the matrix into test and train split with same size of both
    
    np.random.seed(seed)
    test_cut = int((1 - ratio) * 100)
    Xtr = X.copy()
    Xtst = X.copy()
    rated = np.sum(Xtr != 0, axis=1)
    tst = np.around((rated * test_cut) / 100).astype(int)
    for u in range(X.shape[0]):
        idx = np.asarray(np.where(Xtr[u] != 0))[0].tolist()
        idx_tst = np.random.choice(idx, tst[u], replace=False)
        idx_train = list(set(idx).difference(set(idx_tst)))
        Xtr[u, idx_tst] = 0
        Xtst[u, idx_train] = 0
    del idx, idx_train, idx_tst
    return Xtr, Xtst

def _do_stratification(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    is_random=True,
    seed=42,
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    #This function split by each group and aggregate splits together
    
    if not (filter_by == "user" or filter_by == "movie"):
        raise ValueError("filter_by should be either 'user' or 'item'.")
    if min_rating < 1:
        raise ValueError("min_rating should be integer and larger than or equal to 1.")
    if col_user not in data.columns:
        raise ValueError("Schema of data not valid. Missing User Col")
    if col_item not in data.columns:
        raise ValueError("Schema of data not valid. Missing Item Col")
    if not is_random:
        if col_timestamp not in data.columns:
            raise ValueError("Schema of data not valid. Missing Timestamp Col")

    multi_split, ratio = process_split_ratio(ratio)
    split_by_column = col_user if filter_by == "user" else col_item
    ratio = ratio if multi_split else [ratio, 1 - ratio]

    if min_rating > 1:
        data = min_rating_filter_pandas(
            data,
            min_rating=min_rating,
            filter_by=filter_by,
            col_user=col_user,
            col_item=col_item,
        )
    splits = []

    # If it is for chronological splitting, the split will be performed in a random way.
    df_grouped = (
        data.sort_values(col_timestamp).groupby(split_by_column)
        if is_random is False
        else data.groupby(split_by_column)
    )

    for _, group in df_grouped:
        group_splits = split_pandas_data_with_ratios(
            group, ratio, shuffle=is_random, seed=seed
        )

        # Concatenate the list of split dataframes.
        concat_group_splits = pd.concat(group_splits)

        splits.append(concat_group_splits)

    # Concatenate splits for all the groups together.
    splits_all = pd.concat(splits)

    # Take split by split_index
    splits_list = [
        splits_all[splits_all["split_index"] == x].drop("split_index", axis=1)
        for x in range(len(ratio))
    ]

    return splits_list


#This part is copied for ncf checking
def python_chrono_split(
    data,
    ratio=0.75,
    min_rating=1,
    filter_by="user",
    col_user=DEFAULT_USER_COL,
    col_item=DEFAULT_ITEM_COL,
    col_timestamp=DEFAULT_TIMESTAMP_COL,
):
    
    return _do_stratification(
        data,
        ratio=ratio,
        min_rating=min_rating,
        filter_by=filter_by,
        col_user=col_user,
        col_item=col_item,
        col_timestamp=col_timestamp,
        is_random=False,
    )
