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
    DATA_PATH = Path("")
    OOF_DATA_PATH = Path("/workspaces/fuda2_output/oof")
    MODEL_DATA_PATH = Path("/workspaces/fuda2_output/models")
    SUB_DATA_PATH = Path("/workspaces/fuda2_output/submission")
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


def lightgbm_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_DATA_PATH
                / f"lightgbm_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(x_test)
        test_pred += pred
    return test_pred / CFG.n_folds


def xgboost_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_DATA_PATH
                / f"xgboost_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict(xgb.DMatrix(x_test))
        test_pred += pred
    return test_pred / CFG.n_folds


def catboost_inference(x_test: pd.DataFrame):
    test_pred = np.zeros(len(x_test))
    for fold in range(CFG.n_folds):
        model = pickle.load(
            open(
                CFG.MODEL_DATA_PATH
                / f"catboost_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "rb",
            )
        )
        # Predict
        pred = model.predict_proba(x_test)[:, 1]
        test_pred += pred
    return test_pred / CFG.n_folds


def gradient_boosting_model_inference(
    method: str, test_df: pd.DataFrame, features: list, categorical_features: list
):
    x_test = test_df[features]
    if method == "lightgbm":
        test_pred = lightgbm_inference(x_test)
    if method == "xgboost":
        test_pred = xgboost_inference(x_test)
    if method == "catboost":
        test_pred = catboost_inference(x_test)
    return test_pred


def predict(data: pd.DataFrame, features: list, categorical_features: list):
    data = data.copy()
    data["pred_prob"] = 0
    for method in CFG.METHOD_LIST:
        data[f"{method}_pred_prob"] = gradient_boosting_model_inference(
            method, data, features, categorical_features
        )
        data["pred_prob"] += CFG.model_weight_dict[method] * data[f"{method}_pred_prob"]
    return data


def postprocess(
    train_df: pd.DataFrame(), test_df: pd.DataFrame()
) -> (pd.DataFrame(), pd.DataFrame()):
    train_df["pred_prob"] = 0
    for method in CFG.METHOD_LIST:
        oof_df = pd.read_csv(
            CFG.OOF_DATA_PATH / f"oof_{method}_seed{CFG.seed}_ver{CFG.VER}.csv"
        )
        train_df["pred_prob"] += (
            CFG.model_weight_dict[method] * oof_df[f"{method}_prediction"]
        )
    best_score = 0
    best_v = 0
    for v in tqdm(np.arange(1000) / 1000):
        score = f1_score(
            oof_df[CFG.target_col], train_df[f"pred_prob"] >= v, average="macro"
        )
        if score > best_score:
            best_score = score
            best_v = v
    print(best_score, best_v)
    test_df["target"] = np.where(test_df["pred_prob"] >= best_v, 1, 0)
    return train_df, test_df
