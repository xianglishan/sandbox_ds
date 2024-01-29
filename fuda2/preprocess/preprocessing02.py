import numpy as np
import pandas as pd


def preprocess(data: pd.DataFrame()) -> pd.DataFrame():
    print(f"********* もともとのデータshape:{data.shape} *********")

    # $のやつ
    def deal_doller(data: pd.DataFrame) -> pd.DataFrame:
        data["DisbursementGross"] = (
            data["DisbursementGross"]
            .str.replace("$", "")
            .str.replace(",", "")
            .str.replace(" ", "")
            .astype(float)
        )
        data["GrAppv"] = (
            data["GrAppv"]
            .str.replace("$", "")
            .str.replace(",", "")
            .str.replace(" ", "")
            .astype(float)
        )
        data["SBA_Appv"] = (
            data["SBA_Appv"]
            .str.replace("$", "")
            .str.replace(",", "")
            .str.replace(" ", "")
            .astype(float)
        )
        return data

    data = deal_doller(data)

    # いろいろ特徴量作成を追加する
    def make_features(data: pd.DataFrame()) -> pd.DataFrame():
        data = data.copy()
        # 新規ビジネスかどうか
        data["NewExist"] = data["NewExist"].map({1: 0, 2: 1})
        # フランチャイズコードは意味なさそう
        data["FranchiseCode"] = data["FranchiseCode"].apply(
            lambda x: 0 if x in (0, 1) else 1
        )
        # リボルビング信用枠か *Y = はい、N = いいえ
        data["RevLineCr"] = data["RevLineCr"].apply(
            lambda x: 1 if x == "Y" else 0 if x == "N" else np.nan
        )
        # 15 万ドル未満のローンを 1 ページの短い申請で処理できるプログラムか *Y = はい、N = いいえ
        data["LowDoc"] = data["LowDoc"].apply(
            lambda x: 1 if x == "Y" else 0 if x == "N" else np.nan
        )
        # count encoding
        #   Sector 31~33	製造業, 44~45	小売業, 48~49	運輸業、倉庫業 -> one hot
        data["Sector"] = data["Sector"].replace({32: 31, 33: 31, 45: 44, 49: 48})
        for col in ["UrbanRural", "State", "BankState", "City", "Sector"]:
            count_dict = dict(data[col].value_counts())
            data[f"{col}_count_encoding"] = data[col].map(count_dict)
            data[f"{col}_count_encoding"] = (
                data[col].map(count_dict).fillna(1).astype(int)
            )
        return data

    data = make_features(data)

    print(f"********* 加工後のデータshape{data.shape} *********")
    return data
