import numpy as np
import pandas as pd
import os


def read_data():
    train = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/train.csv", index_col=0)
    first_column = train.pop("MIS_Status")
    train.insert(0, "MIS_Status", first_column)  # MIS_Status はターゲットなので先頭にしておく
    test = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/test.csv", index_col=0)
    return (train, test)
