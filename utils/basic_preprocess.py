from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from preprocess.encoding import CustomOneHotEncoding
from preprocess.scaler import CustomScaler
from preprocess.features_selection import BestFeatures
import pandas as pd
from sklearn.decomposition import PCA


def basic_preprocess(df, target, scaler=False, pca=False):
    cat_columns = df.drop(target, axis=1).select_dtypes(include="object").columns.tolist()
    not_cat_columns = df.drop(target, axis=1).select_dtypes(exclude="object").columns.tolist()
    [df_train, unseen_df] = train_test_split(df, test_size=0.2, shuffle=True, random_state=42, stratify=df[target])
    [df_test, df_val]   = train_test_split(unseen_df, test_size=0.5, shuffle=True, random_state=42, stratify=unseen_df[target])

    print(df_train.head(5))
    print(df_val.head(5))
    print(df_test.head(5))

    pipe = Pipeline(steps=[
        ("encoding", CustomOneHotEncoding(cat_columns)),
        #("features_selection", BestFeatures(threshold=0.01, classification=True))
    ])

    X_train = pd.DataFrame(pipe.fit_transform(df_train.drop(target, axis=1), df_train[target]), index=df_train.index)
    X_val   = pd.DataFrame(pipe.transform(df_val.drop(target, axis=1)), index=df_val.index)
    X_test  = pd.DataFrame(pipe.transform(df_test.drop(target, axis=1)), index=df_test.index)

    if scaler or pca:
        steps = []
        if scaler:
            steps.append(
                ("scaler", CustomScaler(not_cat_columns))
            )
        if pca:
            steps.append(
                ("pca", PCA(n_components=0.999))
            )
        pipe = Pipeline(steps=steps)
        X_train = pd.DataFrame(pipe.fit_transform(df_train.drop(target, axis=1)), index=df_train.index)
        X_val   = pd.DataFrame(pipe.transform(df_val.drop(target, axis=1)), index=df_val.index)
        X_test  = pd.DataFrame(pipe.transform(df_test.drop(target, axis=1)), index=df_test.index)

    df_train = pd.concat([X_train, df_train[target]], axis=1)
    df_val = pd.concat([X_val, df_val[target]], axis=1)
    df_test = pd.concat([X_test, df_test[target]], axis=1)


    return [df_train, df_test, df_val]
