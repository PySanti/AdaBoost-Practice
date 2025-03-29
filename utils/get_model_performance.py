from sklearn.metrics import  f1_score
from utils.best_hp import best_hp


def get_model_performance(alg,param_grid, df_train, df_test, target):

    """
        Recibe un algoritmo, encuentra su mejor combinacion de hiperparametros
        , entrena al algoritmo con esa combinacion, y retorna el performance para
        train y test
    """
    alg_instance = best_hp(alg, param_grid, df_train, target)
    train_pred = f1_score(df_train[target], alg_instance.predict(df_train.drop(target, axis=1)), pos_label=1)
    test_pred = f1_score(df_test[target], alg_instance.predict(df_test.drop(target, axis=1)), pos_label=1)

    return [train_pred, test_pred]
