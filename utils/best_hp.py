from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, f1_score

def best_hp(alg, param_grid, df_train, target):
    """
        Recibe la variante de preprocesamiento y retorna la mejor combinacion
        de hiper parametros para cada uno de los algoritmos
    """
    return GridSearchCV(
        alg(),
        param_grid, 
        cv=4,
        n_jobs=5,
        verbose=10,
        scoring=make_scorer(f1_score, pos_label=1)
    ).fit(df_train.drop(target, axis=1), df_train[target]).best_estimator_


