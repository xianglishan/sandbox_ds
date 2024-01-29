import sys

# ====================================================
# Library
# ====================================================
import os
import gc
import warnings

warnings.filterwarnings("ignore")
import random
import scipy as sp
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
import joblib
import pickle
import itertools
from tqdm.auto import tqdm

import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GroupKFold
from sklearn.metrics import log_loss, roc_auc_score, matthews_corrcoef, f1_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import Pool, CatBoostRegressor, CatBoostClassifier


# ====================================================
# Configurations
# ====================================================
class CFG:
    VER = 1
    AUTHOR = "takaito"
    COMPETITION = "FUDA2"
    DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/FUDA2/data")
    OOF_DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/FUDA2/oof")
    MODEL_DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/FUDA2/models")
    SUB_DATA_PATH = Path("/content/drive/MyDrive/Colab Notebooks/FUDA2/submission")
    METHOD_LIST = ["lightgbm", "xgboost", "catboost"]
    seed = 42
    n_folds = 7
    target_col = "MIS_Status"
    metric = "f1_score"
    metric_maximize_flag = True
    num_boost_round = 500
    early_stopping_round = 200
    verbose = 25
    classification_lgb_params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "seed": seed,
    }
    classification_xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "learning_rate": 0.05,
        "random_state": seed,
    }

    classification_cat_params = {
        "learning_rate": 0.05,
        "iterations": num_boost_round,
        "random_seed": seed,
    }
    model_weight_dict = {"lightgbm": 0.50, "xgboost": 0.10, "catboost": 0.40}


# ====================================================
# Read data
# ====================================================
def read_data():
    train = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/train.csv", index_col=0)
    first_column = train.pop("MIS_Status")
    train.insert(0, "MIS_Status", first_column)  # MIS_Status はターゲットなので先頭にしておく
    test = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/test.csv", index_col=0)
    return (train, test)


# ====================================================
# Seed everything
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ====================================================
# Metric
# ====================================================
# f1_score


# ====================================================
# LightGBM Metric
# ====================================================
def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return (
        "f1score",
        f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), average="macro"),
        CFG.metric_maximize_flag,
    )


# ====================================================
# XGBoost Metric
# ====================================================
def xgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return "f1score", f1_score(y_true, np.where(y_pred >= 0.5, 1, 0), average="macro")


def preprocess(data: pd.DataFrame()) -> pd.DataFrame():
    default_numerical_features = [
        "Term",
        "NoEmp",
        "CreateJob",
        "RetainedJob",
        "DisbursementGross",
        "GrAppv",
        "SBA_Appv",
        "ApprovalFY",
    ]
    default_categorical_features = [
        "NewExist",
        "FranchiseCode",
        "RevLineCr",
        "LowDoc",
        "UrbanRural",
        "State",
        "BankState",
        "City",
        "Sector",
    ]
    add_numerical_features = [
        "FranchiseCode_count_encoding",
        "RevLineCr_count_encoding",
        "LowDoc_count_encoding",
        "UrbanRural_count_encoding",
        "State_count_encoding",
        "BankState_count_encoding",
        "City_count_encoding",
        "Sector_count_encoding",
    ]
    numerical_features = add_numerical_features + default_numerical_features
    categorical_features = ["RevLineCr", "LowDoc", "UrbanRural", "State", "Sector"]
    features = numerical_features + categorical_features

    def deal_missing(data: pd.DataFrame()) -> pd.DataFrame():
        data = data.copy()
        for col in ["RevLineCr", "LowDoc", "BankState", "DisbursementDate"]:
            data[col] = data[col].fillna("[UNK]")
        return data

    def clean_money(data: pd.DataFrame()) -> pd.DataFrame():
        data = data.copy()
        for col in ["DisbursementGross", "GrAppv", "SBA_Appv"]:
            data[col] = (
                data[col]
                .str[1:]
                .str.replace(",", "")
                .str.replace(" ", "")
                .astype(float)
            )
        return data

    data = deal_missing(data)
    data = clean_money(data)
    data["NewExist"] = np.where(data["NewExist"] == 1, 1, 0)

    def make_features(data: pd.DataFrame()) -> pd.DataFrame():
        data = data.copy()
        # いろいろ特徴量作成を追加する
        for col in [
            "FranchiseCode",
            "RevLineCr",
            "LowDoc",
            "UrbanRural",
            "State",
            "BankState",
            "City",
            "Sector",
        ]:
            count_dict = dict(data[col].value_counts())
            data[f"{col}_count_encoding"] = data[col].map(count_dict)
            data[f"{col}_count_encoding"] = (
                data[col].map(count_dict).fillna(1).astype(int)
            )

        for col in categorical_features:
            encoder = LabelEncoder()
            encoder.fit(data[col])
            data[col] = encoder.transform(data[col])
        return data

    data = make_features(data)
    return data


if __name__ == "__main__":
    seed_everything(CFG.seed)
    train_df, test_df = read_data()
