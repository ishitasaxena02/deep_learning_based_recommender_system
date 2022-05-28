import numpy as np
import pandas as pd
from functools import wraps
from sklearn.metrics import ( mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, roc_auc_score, log_loss, )
from recommenders.utils.constants import ( DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL, DEFAULT_RELEVANCE_COL, DEFAULT_SIMILARITY_COL, DEFAULT_ITEM_FEATURES_COL, DEFAULT_ITEM_SIM_MEASURE, DEFAULT_K, DEFAULT_THRESHOLD,)
from recommenders.datasets.pandas_df_utils import ( has_columns, has_same_base_dtype, lru_cache_df,)

def _check_column_dtypes(func):
    #Checks columns of DataFrame inputs
    
    @wraps(func)
    def check_column_dtypes_wrapper( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, *args, **kwargs ):
        if not has_columns(rating_true, [col_user, col_item, col_rating]):
            raise ValueError("Missing columns in true rating DataFrame")
        if not has_columns(rating_pred, [col_user, col_item, col_prediction]):
            raise ValueError("Missing columns in predicted rating DataFrame")
        if not has_same_base_dtype(
            rating_true, rating_pred, columns=[col_user, col_item]
        ):
            raise ValueError("Columns in provided DataFrames are not the same datatype")
        return func( rating_true=rating_true, rating_pred=rating_pred, col_user=col_user, col_item=col_item, col_rating=col_rating, col_prediction=col_prediction, *args,  **kwargs )
    return check_column_dtypes_wrapper

@_check_column_dtypes
@lru_cache_df(maxsize=1)

def merge_rating_true_pred( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, ):
    #Join truth and prediction data frames on userID and itemID and return the true and predicted rated with the correct index
    
    suffixes = ["_true", "_pred"]
    rating_true_pred = pd.merge(
        rating_true, rating_pred, on=[col_user, col_item], suffixes=suffixes
    )
    if col_rating in rating_pred.columns:
        col_rating = col_rating + suffixes[0]
    if col_prediction in rating_true.columns:
        col_prediction = col_prediction + suffixes[1]
    return rating_true_pred[col_rating], rating_true_pred[col_prediction]

def rmse( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, ):
    #Calculate Root Mean Squared Error
    
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mae( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, ):
    #Calculate Mean Absolute Error
    
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return mean_absolute_error(y_true, y_pred)

def rsquared( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, ):
    #Calculate R squared
    
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return r2_score(y_true, y_pred)

def exp_var( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, ):
    #Calculate explained variance
    
    y_true, y_pred = merge_rating_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
    )
    return explained_variance_score(y_true, y_pred)

@_check_column_dtypes
@lru_cache_df(maxsize=1)
def merge_ranking_true_pred( rating_true, rating_pred, col_user, col_item, col_rating, col_prediction, relevancy_method, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD,):
    #Filter truth and prediction data frames on common users
    
    common_users = set(rating_true[col_user]).intersection(set(rating_pred[col_user]))
    rating_true_common = rating_true[rating_true[col_user].isin(common_users)]
    rating_pred_common = rating_pred[rating_pred[col_user].isin(common_users)]
    n_users = len(common_users)
    if relevancy_method == "top_k":
        top_k = k
    elif relevancy_method == "by_threshold":
        top_k = threshold
    elif relevancy_method is None:
        top_k = None
    else:
        raise NotImplementedError("Invalid relevancy_method")
    df_hit = get_top_k_items(
        dataframe=rating_pred_common,
        col_user=col_user,
        col_rating=col_prediction,
        k=top_k,
    )
    df_hit = pd.merge(df_hit, rating_true_common, on=[col_user, col_item])[
        [col_user, col_item, "rank"]
    ]
    df_hit_count = pd.merge(
        df_hit.groupby(col_user, as_index=False)[col_user].agg({"hit": "count"}),
        rating_true_common.groupby(col_user, as_index=False)[col_user].agg(
            {"actual": "count"}
        ),
        on=col_user,
    )
    return df_hit, df_hit_count, n_users

def precision_at_k( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, relevancy_method="top_k", k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, ):
    #Precision at K
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred( rating_true=rating_true,rating_pred=rating_pred, col_user=col_user, col_item=col_item, col_rating=col_rating, col_prediction=col_prediction, relevancy_method=relevancy_method, k=k, threshold=threshold, )
    if df_hit.shape[0] == 0:
        return 0.0
    return (df_hit_count["hit"] / k).sum() / n_users

def recall_at_k( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, relevancy_method="top_k", k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, ):
    #Recall at K
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )
    if df_hit.shape[0] == 0:
        return 0.0
    return (df_hit_count["hit"] / df_hit_count["actual"]).sum() / n_users

def ndcg_at_k( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, relevancy_method="top_k", k=DEFAULT_K, threshold=DEFAULT_THRESHOLD,):
    #Calculate Normalized Discounted Cumulative Gain (nDCG)
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )
    if df_hit.shape[0] == 0:
        return 0.0
    df_dcg = df_hit.copy()
    df_dcg["dcg"] = 1 / np.log1p(df_dcg["rank"])
    df_dcg = df_dcg.groupby(col_user, as_index=False, sort=False).agg({"dcg": "sum"})
    df_ndcg = pd.merge(df_dcg, df_hit_count, on=[col_user])
    df_ndcg["idcg"] = df_ndcg["actual"].apply(
        lambda x: sum(1 / np.log1p(range(1, min(x, k) + 1)))
    )
    return (df_ndcg["dcg"] / df_ndcg["idcg"]).sum() / n_users

def map_at_k( rating_true, rating_pred, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_prediction=DEFAULT_PREDICTION_COL, relevancy_method="top_k", k=DEFAULT_K, threshold=DEFAULT_THRESHOLD, ):
   #Mean Average Precision at k 
    
    df_hit, df_hit_count, n_users = merge_ranking_true_pred(
        rating_true=rating_true,
        rating_pred=rating_pred,
        col_user=col_user,
        col_item=col_item,
        col_rating=col_rating,
        col_prediction=col_prediction,
        relevancy_method=relevancy_method,
        k=k,
        threshold=threshold,
    )
    if df_hit.shape[0] == 0:
        return 0.0
    df_hit_sorted = df_hit.copy()
    df_hit_sorted["rr"] = (
        df_hit_sorted.groupby(col_user).cumcount() + 1
    ) / df_hit_sorted["rank"]
    df_hit_sorted = df_hit_sorted.groupby(col_user).agg({"rr": "sum"}).reset_index()
    df_merge = pd.merge(df_hit_sorted, df_hit_count, on=col_user)
    return (df_merge["rr"] / df_merge["actual"]).sum() / n_users

def get_top_k_items( dataframe, col_user=DEFAULT_USER_COL, col_rating=DEFAULT_RATING_COL, k=DEFAULT_K):
    #Get the input customer-item-rating tuple in the format of Pandas DataFrame, output a Pandas DataFrame in the dense format of top k items for each user
    
    if k is None:
        top_k_items = dataframe
    else:
        top_k_items = (
            dataframe.groupby(col_user, as_index=False)
            .apply(lambda x: x.nlargest(k, col_rating))
            .reset_index(drop=True)
        )
    top_k_items["rank"] = top_k_items.groupby(col_user, sort=False).cumcount() + 1
    return top_k_items

metrics = {
    rmse.__name__: rmse,
    mae.__name__: mae,
    rsquared.__name__: rsquared,
    exp_var.__name__: exp_var,
    precision_at_k.__name__: precision_at_k,
    recall_at_k.__name__: recall_at_k,
    ndcg_at_k.__name__: ndcg_at_k,
    map_at_k.__name__: map_at_k,
}


