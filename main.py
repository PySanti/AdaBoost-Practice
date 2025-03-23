import pandas as pd
from scipy.special import basic
from utils.basic_preprocess import basic_preprocess
from utils.generate_filename import generate_filename
from sklearn.ensemble import RandomForestClassifier
from utils.best_hp import best_hp
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from utils.constants import (
    param_grid_bnb,
    param_grid_rf,
    param_grid_lr
)
from sklearn.metrics import make_scorer, f1_score
from utils.get_model_performance import get_model_performance
from utils.get_model_boosted_performance import get_model_boosted_performance




target = "HeartDisease"


for outliers in [True, False]:
    for scaler in [True, False]:
        for pca in [True, False]:
            preprocessing_variant_name = generate_filename(scaler, pca, outliers)
            print("~~~~~~~~~~~")
            print(f"Calculos para variante de preprocesamiento : {preprocessing_variant_name}")
            [df_train, df_test, df_val] = basic_preprocess(pd.read_csv("./data/data.csv"), target,scaler, pca, outliers)
            [nb_best_hp, lr_best_hp] = best_hp(df_train, target)

            [nb_train_performance, nb_test_performance] = get_model_performance(BernoulliNB, df_train, df_test, nb_best_hp, target)
            [lr_train_performance, lr_test_performance] = get_model_performance(LogisticRegression, df_train, df_test, lr_best_hp, target)
            [nb_b_train_performance, nb_b_test_performance] = get_model_boosted_performance(BernoulliNB, df_train, df_test, nb_best_hp, target)
            [lr_b_train_performance, lr_b_test_performance] = get_model_boosted_performance(LogisticRegression, df_train, df_test, lr_best_hp, target)



            results = {
                "naive bayes" : {
                    "hp" : nb_best_hp, 
                    "train_p" : nb_train_performance, 
                    "test_p" : nb_test_performance, 
                    "train_p_b" : nb_b_train_performance, 
                    "test_p_b" : nb_b_test_performance, 
                },
                "regresion logistica " : {
                    "hp" : lr_best_hp, 
                    "train_p" : lr_train_performance, 
                    "test_p" : lr_test_performance,
                    "train_p_b" : lr_b_train_performance,
                    "test_p_b" : lr_b_test_performance

                },
            }
            with open("./results.txt", "a") as f:
                f.write(f"Variante de preprocesamiento : {preprocessing_variant_name}\n")
                for k,v in results.items():
                    f.write(f"{k}\n")
                    f.write(f"\tMejor combinacion de hiperparametros : {v['hp']}\n")
                    f.write(f"\tPerformance en train : {v['train_p']:.2f}\n")
                    f.write(f"\tPerformance en test : {v['test_p']:.2f}\n")
                    f.write(f"\tPerformance boosted en train : {v['train_p_b']:.2f}\n")
                    f.write(f"\tPerformance boosted en test : {v['test_p_b']:.2f}\n")
