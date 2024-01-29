import os
import sys
from pycaret.classification import *

from preprocess.preprocessing03 import preprocess
from read_data.read import read_data

# from learn.metric_mean_f1 import score
from learn.learning01 import learn
from learn.predict01 import predict, postprocess

import subprocess


def main():
    # read data
    train, test = read_data()

    # preprocess
    preped_train = preprocess(train)
    preped_test = preprocess(test)

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

    # learn
    learn(
        data=preped_train, features=features, categorical_features=categorical_features
    )

    # predict
    predicted = predict(
        data=preped_test, features=features, categorical_features=categorical_features
    )

    # postprocess
    train_out, test_out = postprocess(preped_train, predicted)

    # submission
    test_out[["target"]].to_csv(
        "../../fuda2_output/submission/submission.csv", header=False
    )
    subprocess.call(
        "signate submit --competition-id 1337 ../../fuda2_output/submission/submission.csv"
    )


if __name__ == "__main__":
    main()
