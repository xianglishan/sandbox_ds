import pandas as pd
from ydata_profiling import ProfileReport

data = pd.read_csv("s3://xianglishan-sandbox/signate-fuda2/train.csv")
profile = ProfileReport(data, title="Profiling Report")
# profile.to_notebook_iframe()


profile.to_file("./train_data_report.html")

str = "hoge" "fuga" "piyo"
