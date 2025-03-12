from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler
from preprocess.features_selection import BestFeatures
import pandas as pd
from sklearn.decomposition import PCA
from preprocess.outliers_info import outliers_info

def show_results(df_list : list):
    for d in df_list:
        print(d.head(2))
    print("________________________________________")

def basic_preprocess(df, target, scaler=False, pca=False, outliers=False):
    cat_columns = df.drop(target, axis=1).select_dtypes(include="object").columns.tolist()
    not_cat_columns = df.drop(target, axis=1).select_dtypes(exclude="object").columns.tolist()

    [df_train, unseen_df] = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df[target])
    [df_test, df_val]   = train_test_split(unseen_df, test_size=0.5, shuffle=True, random_state=42, stratify=unseen_df[target])

    pipe = Pipeline(steps=[
        ("encoding", CustomOneHotEncoding(cat_columns)),
        ("features_selection", BestFeatures(threshold=0.01, classification=True))
    ])

    show_results([df_train, df_val, df_test])

    X_train = pd.DataFrame(pipe.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
    X_val   = pd.DataFrame(pipe.transform(df_val.drop(target, axis=1)), index=df_val.index)
    X_test  = pd.DataFrame(pipe.transform(df_test.drop(target, axis=1)), index=df_test.index)

    show_results([X_train, X_val, X_test])

    if scaler or pca:
        steps = []
        if scaler: steps.append(("scaler", CustomScaler(not_cat_columns)))
        if pca: steps.append(("pca", PCA(n_components=0.999)))
        pipe = Pipeline(steps=steps)
        X_train = pd.DataFrame(pipe.fit_transform(X_train), index=X_train.index)
        X_val   = pd.DataFrame(pipe.transform(X_val), index=X_val.index)
        X_test  = pd.DataFrame(pipe.transform(X_test), index=X_test.index)

    if outliers:
        len_1 = len(X_train)
        outliers_indexes = outliers_info(X_train).query("outliers_ == -1").index
        X_train = X_train.drop(outliers_indexes)
        df_train = df_train.drop(outliers_indexes)
        print(f"Deleted samples : {len_1 - len(X_train)}/{len_1} ({(len_1 - len(X_train))*100/len_1})")


    show_results([X_train, X_val, X_test])

    df_train = pd.concat([X_train, df_train[target]], axis=1)
    df_val = pd.concat([X_val, df_val[target]], axis=1)
    df_test = pd.concat([X_test, df_test[target]], axis=1)




    return [df_train, df_test, df_val]
