import os
import re
import random
import shutil
import warnings
import pandas as pd
from typing import Optional
from zipfile import ZipFile
from recommenders.datasets.download_utils import maybe_download, download_path
from recommenders.utils.notebook_utils import is_databricks
from recommenders.utils.constants import ( DEFAULT_HEADER, DEFAULT_ITEM_COL,  DEFAULT_USER_COL, DEFAULT_RATING_COL,  DEFAULT_TIMESTAMP_COL, DEFAULT_TITLE_COL, DEFAULT_GENRE_COL,
)
try:
    from pyspark.sql.types import (
        StructType, StructField, StringType, IntegerType, FloatType, LongType,
    )
except ImportError:
    pass
import pandera as pa
import pandera.extensions as extensions
from pandera import Field
from pandera.typing import Series

class _DataFormat:
    #MovieLens data format container as a different size of MovieLens data file has a different format
    
    def __init__( self, sep, path, has_header=False, item_sep=None, item_path=None, item_has_header=False, ):
        
        #Rating file
        self._sep = sep
        self._path = path
        self._has_header = has_header
        
        #Item file
        self._item_sep = item_sep
        self._item_path = item_path
        self._item_has_header = item_has_header
        
    @property
    def separator(self):
        return self._sep
    @property
    def path(self):
        return self._path
    @property
    def has_header(self):
        return self._has_header
    @property
    def item_separator(self):
        return self._item_sep
    @property
    def item_path(self):
        return self._item_path
    @property
    def item_has_header(self):
        return self._item_has_header
    
#10m and 20m data do not have user data    
    
DATA_FORMAT = {
    "100k": _DataFormat("\t", "ml-100k/u.data", False, "|", "ml-100k/u.item", False),
    "1m": _DataFormat(
        "::", "ml-1m/ratings.dat", False, "::", "ml-1m/movies.dat", False
    ),
    "10m": _DataFormat(
        "::", "ml-10M100K/ratings.dat", False, "::", "ml-10M100K/movies.dat", False
    ),
    "20m": _DataFormat(",", "ml-20m/ratings.csv", True, ",", "ml-20m/movies.csv", True),
}

#Fake data for testing only
MOCK_DATA_FORMAT = {
    "mock100": {"size": 100, "seed": 6},
}

#100K data genres index to string mapper. For 1m, 10m, and 20m, the genres labels are already in the dataset.
GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",  "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)

#Various warnings
WARNING_MOVIE_LENS_HEADER = """MovieLens rating dataset has four columns
    (user id, movie id, rating, and timestamp), but more than four column names are provided.
    Will only use the first four column names."""
WARNING_HAVE_SCHEMA_AND_HEADER = """Both schema and header are provided.
    The header argument will be ignored."""
ERROR_MOVIE_LENS_SIZE = (
    "Invalid data size. Should be one of {100k, 1m, 10m, or 20m, or mock100}"
)
ERROR_HEADER = "Header error. At least user and movie column names should be provided"

def load_pandas_df( size="100k", header=None, local_cache_path=None, title_col=None, genres_col=None, year_col=None,):
    #Loads the MovieLens dataset as pd.DataFrame
    #Download the dataset from https://files.grouplens.org/datasets/movielens, unzip, and load
    
    size = size.lower()
    if size not in DATA_FORMAT and size not in MOCK_DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)
    if header is None:
        header = DEFAULT_HEADER
    elif len(header) < 2:
        raise ValueError(ERROR_HEADER)
    elif len(header) > 4:
        warnings.warn(WARNING_MOVIE_LENS_HEADER)
        header = header[:4]
    if size in MOCK_DATA_FORMAT:
        return MockMovielensSchema.get_df(
            keep_first_n_cols=len(header),
            keep_title_col=(title_col is not None),
            keep_genre_col=(genres_col is not None),
            **MOCK_DATA_FORMAT[size],
        )
    movie_col = header[1]
    with download_path(local_cache_path) as path:
        filepath = os.path.join(path, "ml-{}.zip".format(size))
        datapath, item_datapath = _maybe_download_and_extract(size, filepath)
        item_df = _load_item_df(
            size, item_datapath, movie_col, title_col, genres_col, year_col
        )
        df = pd.read_csv(
            datapath,
            sep=DATA_FORMAT[size].separator,
            engine="python",
            names=header,
            usecols=[*range(len(header))],
            header=0 if DATA_FORMAT[size].has_header else None,
        )
        if len(header) > 2:
            df[header[2]] = df[header[2]].astype(float)
        if item_df is not None:
            df = df.merge(item_df, on=header[1])
    return df

def _load_item_df(size, item_datapath, movie_col, title_col, genres_col, year_col):
    #Loads Movie info
    
    if title_col is None and genres_col is None and year_col is None:
        return None
    item_header = [movie_col]
    usecols = [0]
    
    # Year is parsed from title
    if title_col is not None or year_col is not None:
        item_header.append("title_year")
        usecols.append(1)

    genres_header_100k = None
    if genres_col is not None:
        # 100k data's movie genres are encoded as a binary array (the last 19 fields)
        
        if size == "100k":
            genres_header_100k = [*(str(i) for i in range(19))]
            item_header.extend(genres_header_100k)
            usecols.extend([*range(5, 24)])
        else:
            item_header.append(genres_col)
            usecols.append(2)

    item_df = pd.read_csv(
        item_datapath,
        sep=DATA_FORMAT[size].item_separator,
        engine="python",
        names=item_header,
        usecols=usecols,
        header=0 if DATA_FORMAT[size].item_has_header else None,
        encoding="ISO-8859-1",
    )

    # Convert 100k data's format: '0|0|1|...' to 'Action|Romance|..."
    if genres_header_100k is not None:
        item_df[genres_col] = item_df[genres_header_100k].values.tolist()
        item_df[genres_col] = item_df[genres_col].map(
            lambda l: "|".join([GENRES[i] for i, v in enumerate(l) if v == 1])
        )

        item_df.drop(genres_header_100k, axis=1, inplace=True)

    # Parse year from movie title. Note, MovieLens title format is "title (year)"
    if year_col is not None:
        def parse_year(t):
            parsed = re.split("[()]", t)
            if len(parsed) > 2 and parsed[-2].isdecimal():
                return parsed[-2]
            else:
                return None
        item_df[year_col] = item_df["title_year"].map(parse_year)
        if title_col is None:
            item_df.drop("title_year", axis=1, inplace=True)
    if title_col is not None:
        item_df.rename(columns={"title_year": title_col}, inplace=True)
    return item_df

def _maybe_download_and_extract(size, dest_path):
    #Downloads and extracts MovieLens rating and item datafiles if they don’t already exist
    
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    _, rating_filename = os.path.split(DATA_FORMAT[size].path)
    rating_path = os.path.join(dirs, rating_filename)
    _, item_filename = os.path.split(DATA_FORMAT[size].item_path)
    item_path = os.path.join(dirs, item_filename)
    if not os.path.exists(rating_path) or not os.path.exists(item_path):
        download_movielens(size, dest_path)
        extract_movielens(size, rating_path, item_path, dest_path)
    return rating_path, item_path

def download_movielens(size, dest_path):
    #Downloads MovieLens datafile
    
    if size not in DATA_FORMAT:
        raise ValueError(ERROR_MOVIE_LENS_SIZE)
    url = "https://files.grouplens.org/datasets/movielens/ml-" + size + ".zip"
    dirs, file = os.path.split(dest_path)
    maybe_download(url, file, work_directory=dirs)


def extract_movielens(size, rating_path, item_path, zip_path):
    #Extract MovieLens rating and item datafiles from the MovieLens raw zip file

    with ZipFile(zip_path, "r") as z:
        with z.open(DATA_FORMAT[size].path) as zf, open(rating_path, "wb") as f:
            shutil.copyfileobj(zf, f)
        with z.open(DATA_FORMAT[size].item_path) as zf, open(item_path, "wb") as f:
            shutil.copyfileobj(zf, f)