import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder,TargetEncoder
from sklearn.compose import ColumnTransformer

class ProblemsTransformer(BaseEstimator, TransformerMixin):
    """
    与えられたDataFrameに対して、problemsを分割して、各問題が含まれているかどうかを特徴量として追加する

    Parameters
    ----------
    raw_df : pd.DataFrame
        problemsを含むDataFrame
    problem_unique : set
        problemsに含まれる問題の集合(testデータを処理する場合には、trainデータの問題の集合を渡す)
    Returns
    -------
    pd.DataFrame
        problemsを分割して、各問題が含まれているかどうかを特徴量として追加したDataFrame
    problem_unique : set
        problemsに含まれる問題の集合(testデータを処理する場合にはNoneを返す)
    """
    def _split_string_by_uppercase(self, input_string):
        result = []
        current_word = ""
        if not type(input_string)==str:
            return result

        # Escape String "Other"
        for char in input_string:
            if char.isupper() and current_word:
                result.append(current_word)
                current_word = char
            else:
                current_word += char

        if current_word:
            result.append(current_word)

        return result
    def fit(self, x_train:pd.DataFrame, problem_unique:set=None) -> list:
        df = x_train.copy()
        df.loc[:,"problems_list"] = df.problems.apply(self._split_string_by_uppercase)
        problem_unique = set()
        for problems in df.loc[:,"problems"].unique():
            for problem in self._split_string_by_uppercase(problems):
                problem_unique.add(problem)
        self.problem_unique = problem_unique
        return self
    def transform(self, x:pd.DataFrame)->pd.DataFrame:
        df = x.copy()
        df.loc[:,"problems_list"] = df.problems.apply(self._split_string_by_uppercase)
        for unique_problem in self.problem_unique:
            df.loc[:,f"is_problem_{unique_problem}"] = df.loc[:,"problems_list"].apply(lambda x: unique_problem in x)
            df.loc[:,f"is_problem_{unique_problem}"] = df.loc[:,f"is_problem_{unique_problem}"].fillna(False)
            df.loc[:,f"is_problem_{unique_problem}"] = df.loc[:,f"is_problem_{unique_problem}"].astype(int)
        df.loc[:,"problems_count"] = df.loc[:,"problems_list"].apply(len).astype(int)
        df = df.drop(columns=["problems_list", "problems"])
        self.columns = df.columns.tolist()
        return df
    def get_feature_names_out(self, input_features=None):
        return self.columns


class SpcEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.spc_latin_df = pd.read_csv("data/additional/spc_latin.csv")
        self.enc = OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    def fit(self, X, y=None):
        df = X.copy()
        df = pd.merge(df, self.spc_latin_df, on="spc_latin", how="left")
        self.enc.fit(df)
        return self
    def transform(self, X):
        df = X.copy()
        df = pd.merge(df, self.spc_latin_df, on="spc_latin", how="left")
        self.columns = df.columns.tolist()
        df = self.enc.transform(df)
        return df
    def get_feature_names_out(self, input_features=None):
        return self.columns


class CyclicEncoder(BaseEstimator, TransformerMixin):
    def cyclic_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This func adds YMDs and those cyclic columns to DataFrame: Year, Month, Day, Dayofweek.

        Usage
        ------------------
        df = cyclic_encode(df)
        """
        df['created_at'] = pd.to_datetime(df['created_at'])  # cast str to datetime64 for a column: 'created_at'
        # make new feature columns
        df['season'] = df['created_at'].dt.quarter  # 季節（：1~4）
        df['year'] = df['created_at'].dt.year  # 年（：2015, 2016）
        df['month'] = df['created_at'].dt.month  # 月（：1~12）
        df['day'] = df['created_at'].dt.day  # その月の日付（：1~31）
        df['dayofweek'] = df['created_at'].dt.dayofweek  # その週の日付（：0~6）
        # make new cyclic feature columns
        cyclic_cols = {
                'season': 4,
                'month': 12,
                'day': 31,
                'dayofweek': 7,
                }
        for col, max_val in cyclic_cols.items():
            df[col + '_cos'] = np.cos(2 * np.pi * df[col] / max_val)
            df[col + '_sin'] = np.sin(2 * np.pi * df[col] / max_val)
        df['month_day_sin'] = np.sin(2 * np.pi * df['month'] / 12 + 2 * np.pi * df['day'] / 31)
        df['month_day_cos'] = np.cos(2 * np.pi * df['month'] / 12 + 2 * np.pi * df['day'] / 31)
        # remove 'created_at' column
        df = df.drop(columns=['created_at', 'year', 'month', 'day'])
        return df
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = self.cyclic_encode(X)
        self.columns = X.columns.tolist()
        return X
    def get_feature_names_out(self, input_features=None):
        return self.columns


def data_augmentation(x_train_raw:pd.DataFrame, y_train_raw:pd.Series) -> (pd.DataFrame, pd.Series):
    """
    This func augments data by SMOTE.
    input required: processed train data
    output: augmented train data
    """
    x_train = x_train_raw.copy()
    y_train = y_train_raw.copy()
    print("Before SMOTE: ", len(x_train))
    sm = SMOTE(
        random_state=42,
        k_neighbors=5,
        n_jobs=-1,
        )
    x_train_auged, y_train_auged = sm.fit_resample(x_train, y_train)
    print("After SMOTE: ", len(x_train_auged))
    return x_train_auged, y_train_auged


def Preprocess(config=None):
    object_columns = config["object_columns"]
    if config["is_target_encode"]:
        ct = ColumnTransformer([
            ('num', 'passthrough', ['tree_dbh']),
            ('problems',ProblemsTransformer(),['problems']),
            ('time',CyclicEncoder(),['created_at']),
            ('cat',SpcEncoder(),['spc_latin']),
            ('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),object_columns),
            ('tar',TargetEncoder(target_type='continuous',random_state=42),object_columns)
        ],remainder='drop')
    else:
        ct = ColumnTransformer([
            ('num', 'passthrough', ['tree_dbh']),
            ('problems',ProblemsTransformer(),['problems']),
            ('time',CyclicEncoder(),['created_at']),
            ('cat_',SpcEncoder(),['spc_latin']),
            ('cat',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1),object_columns),
        ],remainder='drop')
    return ct