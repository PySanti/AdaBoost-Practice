from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score


from utils.constants import (
    param_grid_bnb,
    param_grid_rf,
    param_grid_lr
)

base_grid_config = {
    "cv":4,
    "n_jobs":5,
    "verbose": 10,
    "scoring":make_scorer(f1_score,pos_label=1)
}

def best_hp(df_train, target):
    """
        Recibe la variante de preprocesamiento y retorna la mejor combinacion
        de hiper parametros para cada uno de los algoritmos
    """

    nb = GridSearchCV(
        BernoulliNB(),
        param_grid_bnb, 
        **base_grid_config
    ).fit(df_train.drop(target, axis=1), df_train[target]).best_params_

    lr = GridSearchCV(
        LogisticRegression(),
        param_grid_lr, 
        **base_grid_config
    ).fit(df_train.drop(target, axis=1), df_train[target]).best_params_

    return [nb, lr]
