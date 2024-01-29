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


def Predicting(data: pd.DataFrame, features: list, categorical_features: list):
    data = data.copy()
    data["pred_prob"] = 0
    for method in CFG.METHOD_LIST:
        data[f"{method}_pred_prob"] = gradient_boosting_model_inference(
            method, data, features, categorical_features
        )
        data["pred_prob"] += CFG.model_weight_dict[method] * data[f"{method}_pred_prob"]
    return data
