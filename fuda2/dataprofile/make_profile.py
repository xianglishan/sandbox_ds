import pandas as pd
from ydata_profiling import ProfileReport

train = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/train.csv", index_col=0)
first_column = train.pop("MIS_Status")
train.insert(0, "MIS_Status", first_column)  # MIS_Status はターゲットなので先頭にしておく
train_profile = ProfileReport(train, title="Train Profiling Report")
# profile.to_notebook_iframe()
train_profile.to_file("./train_data_report.html")

test = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/test.csv", index_col=0)
test_profile = ProfileReport(test, title="Test Profiling Report")
# profile.to_notebook_iframe()
test_profile.to_file("./test_data_report.html")
