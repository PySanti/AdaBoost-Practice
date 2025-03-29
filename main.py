import pandas as pd
from utils.basic_preprocess import basic_preprocess
from sklearn.naive_bayes import BernoulliNB
from utils.get_model_performance import get_model_performance
from sklearn.svm import SVC
from utils.constants import param_grid_bnb, param_grid_lr, param_grid_b_lr, param_grid_b_bnb, param_grid_fs, param_grid_rf, param_grid_svc
from sklearn.ensemble import RandomForestClassifier
target = "HeartDisease"




[df_train, df_test] = basic_preprocess(pd.read_csv("./data/data.csv"), target)
print(df_train[target].value_counts())

results = {
    "nb" : {},
    "rf" : {},
    "svc" : {}
    #"lr" : {},
    #"nbb":{},
    #"lrb":{},
    #"fs" : {}
}
[results["nb"]["train"], results["nb"]["test"]] = get_model_performance(BernoulliNB, param_grid_bnb, df_train, df_test, target)
[results["rf"]["train"], results["rf"]["test"]] = get_model_performance(RandomForestClassifier, param_grid_rf, df_train, df_test, target)
[results["svc"]["train"], results["svc"]["test"]] = get_model_performance(SVC, param_grid_svc, df_train, df_test, target)
#[results["lr"]["train"], results["lr"]["test"]] = get_model_performance(LogisticRegression,param_grid_lr,df_train,df_test,target)
#[results["nbb"]["train"], results["nbb"]["test"]] = get_model_performance(AdaBoostClassifier,param_grid_b_bnb,df_train,df_test,target)
#[results["lrb"]["train"], results["lrb"]["test"]] = get_model_performance(AdaBoostClassifier,param_grid_b_lr,df_train,df_test,target)
#[results["fs"]["train"], results["fs"]["test"]] = get_model_performance(AdaBoostClassifier,param_grid_fs,df_train,df_test,target)

with open("./results3.txt", "a") as f:
    for k,v in results.items():
        f.write(f"{k}\n")
        f.write(f"\tPerformance en train : {v['train']:.2f}\n")
        f.write(f"\tPerformance en test : {v['test']:.2f}\n")
