import pandas as pd
import numpy as np
import itertools
from scipy.sparse import coo_matrix
import logging
from recommenders.utils.constants import ( DEFAULT_USER_COL, DEFAULT_ITEM_COL, DEFAULT_RATING_COL, DEFAULT_PREDICTION_COL, )

log = logging.getLogger(__name__)

class AffinityMatrix:
    
    #Generate the user/item affinity matrix from a pandas dataframe and vice versa
    
    def __init__( self, df, items_list=None, col_user=DEFAULT_USER_COL, col_item=DEFAULT_ITEM_COL, col_rating=DEFAULT_RATING_COL, col_pred=DEFAULT_PREDICTION_COL, save_path=None, ):
        self.df = df 
        self.items_list = items_list 
        self.col_item = col_item
        self.col_user = col_user
        self.col_rating = col_rating
        self.col_pred = col_pred
        self.save_path = save_path

    def _gen_index(self):
        # Generate the user/item index
        
        self.df_ = self.df.sort_values(by=[self.col_user])
        unique_users = self.df_[self.col_user].unique()
        if self.items_list is not None:
            unique_items = self.items_list 
        else:
            unique_items = self.df_[
                self.col_item
            ].unique()
        self.Nusers = len(unique_users)
        self.Nitems = len(unique_items)
        self.map_users = {x: i for i, x in enumerate(unique_users)}
        self.map_items = {x: i for i, x in enumerate(unique_items)}
        self.map_back_users = {i: x for i, x in enumerate(unique_users)}
        self.map_back_items = {i: x for i, x in enumerate(unique_items)}
        self.df_.loc[:, "hashedItems"] = self.df_[self.col_item].map(self.map_items)
        self.df_.loc[:, "hashedUsers"] = self.df_[self.col_user].map(self.map_users)
        if self.save_path is not None:
            np.save(self.save_path + "/user_dict", self.map_users)
            np.save(self.save_path + "/item_dict", self.map_items)
            np.save(self.save_path + "/user_back_dict", self.map_back_users)
            np.save(self.save_path + "/item_back_dict", self.map_back_items)

    def gen_affinity_matrix(self):
        #Generate the user/item affinity matrix
        
        log.info("Generating the user/item affinity matrix...")
        self._gen_index()
        ratings = self.df_[self.col_rating]
        itm_id = self.df_["hashedItems"]
        usr_id = self.df_["hashedUsers"]
        self.AM = coo_matrix(
            (ratings, (usr_id, itm_id)), shape=(self.Nusers, self.Nitems)
        ).toarray()
        zero = (self.AM == 0).sum()
        total = self.AM.shape[0] * self.AM.shape[1]
        sparsness = zero / total * 100
        log.info("Matrix generated, sparseness percentage: %d" % sparsness)
        print("Matrix generated, sparseness percentage: %d" % sparsness)
        return self.AM, self.map_users, self.map_items

    def map_back_sparse(self, X, kind):
        #Map back the user/affinity matrix to a pd dataframe
        
        m, n = X.shape
        items = [np.asanyarray(np.where(X[i, :] != 0)).flatten() for i in range(m)]
        ratings = [X[i, items[i]] for i in range(m)]
        userids = []
        for i in range(0, m):
            userids.extend([i] * len(items[i]))
        items = list(itertools.chain.from_iterable(items))
        ratings = list(itertools.chain.from_iterable(ratings))
        if kind == "ratings":
            col_out = self.col_rating
        else:
            col_out = self.col_pred
        out_df = pd.DataFrame.from_dict(
            {self.col_user: userids, self.col_item: items, col_out: ratings}
        )
        out_df[self.col_user] = out_df[self.col_user].map(self.map_back_users)
        out_df[self.col_item] = out_df[self.col_item].map(self.map_back_items)
        return out_df