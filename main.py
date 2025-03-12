import pandas as pd
from utils.basic_preprocess import basic_preprocess
from utils.generate_filename import generate_filename



target = "HeartDisease"

for outliers in [True, False]:
    for scaler in [True, False]:
        for pca in [True, False]:
            [df_train, df_test, df_val] = basic_preprocess(
                pd.read_csv("./data/data.csv"), 
                target, 
                outliers=outliers, 
                scaler=scaler, 
                pca=pca)

