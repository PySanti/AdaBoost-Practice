from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import AdaBoostClassifier

def get_model_boosted_performance(alg, df_train, df_test, best_hp, target):

    """
        Recibe la mejor combinacion de hiperparametros para un
        algoritmo y retorna su performance para el conjunto de entrenamiento y test
    """

    alg_instance = AdaBoostClassifier(
        estimator=alg(),
        n_estimators=100,  # Número de iteradores (estimadores débiles a combinar)
        random_state=42
    )
    alg_instance.fit(df_train.drop(target, axis=1), df_train[target])
    train_pred = f1_score(df_train[target], alg_instance.predict(df_train.drop(target, axis=1)), pos_label=1)
    test_pred = f1_score(df_test[target], alg_instance.predict(df_test.drop(target, axis=1)), pos_label=1)

    return [train_pred, test_pred]
