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


def lightgbm_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    features: list,
    categorical_features: list,
):
    lgb_train = lgb.Dataset(x_train, y_train, categorical_feature=categorical_features)
    lgb_valid = lgb.Dataset(x_valid, y_valid, categorical_feature=categorical_features)
    model = lgb.train(
        params=CFG.classification_lgb_params,
        train_set=lgb_train,
        num_boost_round=CFG.num_boost_round,
        valid_sets=[lgb_train, lgb_valid],
        feval=lgb_metric,
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=CFG.early_stopping_round, verbose=CFG.verbose
            )
        ],
    )
    # Predict validation
    valid_pred = model.predict(x_valid)
    return model, valid_pred


def xgboost_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    features: list,
    categorical_features: list,
):
    xgb_train = xgb.DMatrix(data=x_train, label=y_train)
    xgb_valid = xgb.DMatrix(data=x_valid, label=y_valid)
    model = xgb.train(
        CFG.classification_xgb_params,
        dtrain=xgb_train,
        num_boost_round=CFG.num_boost_round,
        evals=[(xgb_train, "train"), (xgb_valid, "eval")],
        early_stopping_rounds=CFG.early_stopping_round,
        verbose_eval=CFG.verbose,
        feval=xgb_metric,
        maximize=CFG.metric_maximize_flag,
    )
    # Predict validation
    valid_pred = model.predict(xgb.DMatrix(x_valid))
    return model, valid_pred


def catboost_training(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    features: list,
    categorical_features: list,
):
    cat_train = Pool(data=x_train, label=y_train, cat_features=categorical_features)
    cat_valid = Pool(data=x_valid, label=y_valid, cat_features=categorical_features)
    model = CatBoostClassifier(**CFG.classification_cat_params)
    model.fit(
        cat_train,
        eval_set=[cat_valid],
        early_stopping_rounds=CFG.early_stopping_round,
        verbose=CFG.verbose,
        use_best_model=True,
    )
    # Predict validation
    valid_pred = model.predict_proba(x_valid)[:, 1]
    return model, valid_pred


def gradient_boosting_model_cv_training(
    method: str, train_df: pd.DataFrame, features: list, categorical_features: list
):
    # Create a numpy array to store out of folds predictions
    oof_predictions = np.zeros(len(train_df))
    oof_fold = np.zeros(len(train_df))
    kfold = KFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
    for fold, (train_index, valid_index) in enumerate(kfold.split(train_df)):
        print("-" * 50)
        print(f"{method} training fold {fold+1}")

        x_train = train_df[features].iloc[train_index]
        y_train = train_df[CFG.target_col].iloc[train_index]
        x_valid = train_df[features].iloc[valid_index]
        y_valid = train_df[CFG.target_col].iloc[valid_index]
        if method == "lightgbm":
            model, valid_pred = lightgbm_training(
                x_train, y_train, x_valid, y_valid, features, categorical_features
            )
        if method == "xgboost":
            model, valid_pred = xgboost_training(
                x_train, y_train, x_valid, y_valid, features, categorical_features
            )
        if method == "catboost":
            model, valid_pred = catboost_training(
                x_train, y_train, x_valid, y_valid, features, categorical_features
            )

        # Save best model
        pickle.dump(
            model,
            open(
                CFG.MODEL_DATA_PATH
                / f"{method}_fold{fold + 1}_seed{CFG.seed}_ver{CFG.VER}.pkl",
                "wb",
            ),
        )
        # Add to out of folds array
        oof_predictions[valid_index] = valid_pred
        oof_fold[valid_index] = fold + 1
        del x_train, x_valid, y_train, y_valid, model, valid_pred
        gc.collect()

    # Compute out of folds metric
    score = f1_score(train_df[CFG.target_col], oof_predictions >= 0.5, average="macro")
    print(f"{method} our out of folds CV f1score is {score}")
    # Create a dataframe to store out of folds predictions
    oof_df = pd.DataFrame(
        {
            CFG.target_col: train_df[CFG.target_col],
            f"{method}_prediction": oof_predictions,
            "fold": oof_fold,
        }
    )
    oof_df.to_csv(
        CFG.OOF_DATA_PATH / f"oof_{method}_seed{CFG.seed}_ver{CFG.VER}.csv", index=False
    )


def learn(data: pd.DataFrame, features: list, categorical_features: list):
    for method in CFG.METHOD_LIST:
        gradient_boosting_model_cv_training(
            method, data, features, categorical_features
        )
