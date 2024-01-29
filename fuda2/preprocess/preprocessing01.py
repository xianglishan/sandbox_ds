import numpy as np
import pandas as pd


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    print(f"********* もともとのデータshape:{data.shape} *********")

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
    # Sector 31~33	製造業, 44~45	小売業, 48~49	運輸業、倉庫業
    data["Sector"] = data["Sector"].map({32: 31, 33: 31, 45: 44, 49: 48})
    data = pd.get_dummies(data, columns=["Sector"])
    # City
    MIS_City_list = [
        "BLACKLICK",
        "SCOTTSDALE",
        "SAN FRANCISCO",
        "RALEIGH",
        "HIALEAH",
        "BRUNSWICK",
        "ESCONDIDO",
        "CLARENCE",
        "REHOBOTH",
        "BRAWLEY",
    ]
    data["City"] = data["City"].apply(lambda x: x if x in MIS_City_list else np.nan)
    data = pd.get_dummies(data, columns=["City"])
    # Stateは削除しちゃう
    data = data.drop(["State"], axis=1)
    # $のやつ
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
    # UrbanRural *1 = 都市部、2 = 田舎、0 = 未定義, 都会っぽい順に大きくしてみる
    data["UrbanRural"] = data["UrbanRural"].map({1: 2, 2: 0, 0: 1})

    print(f"********* 加工後のデータshape{data.shape} *********")
    return data
