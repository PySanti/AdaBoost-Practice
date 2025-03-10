import pandas as pd
from utils.basic_preprocess import basic_preprocess


target = "HeartDisease"
[df_train, df_val, df_test] = basic_preprocess(pd.read_csv("./data/data.csv"), "")
