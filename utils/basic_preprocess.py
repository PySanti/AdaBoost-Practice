from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.pipeline import Pipeline
from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler
from preprocess.features_selection import BestFeatures
import pandas as pd
from preprocess.outliers_info import outliers_info

def show_results(df_list : list):
    for d in df_list:
        print(d.head(2))
    print("________________________________________")

def basic_preprocess(df, target):

    df[target] = df[target].map({"Yes":1, "No" : 0})

    cat_columns = df.drop(target, axis=1).select_dtypes(include="object").columns.tolist()
    not_cat_columns = df.drop(target, axis=1).select_dtypes(exclude="object").columns.tolist()

    [df_train, df_test] = train_test_split(df, test_size=0.3, shuffle=True, random_state=42, stratify=df[target])

    pipe = Pipeline(steps=[
        ("encoding", CustomOneHotEncoding(cat_columns)),
        ("features_selection", BestFeatures(threshold=0.01, classification=True)),
        ("scaler", CustomScaler(not_cat_columns))
    ])
    X_train = pd.DataFrame(pipe.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
    X_test  = pd.DataFrame(pipe.transform(df_test.drop(target, axis=1)), index=df_test.index)

    X_train, y_resampled = ADASYN(random_state=42).fit_resample(X_train, df_train[target])

    df_train = pd.concat([X_train, y_resampled], axis=1)
    df_test = pd.concat([X_test, df_test[target]], axis=1)

    return [df_train, df_test]
