import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier

param_grid_rf = {  
    'n_estimators': np.arange(50, 351, 50),
    'max_features': ['sqrt', 'log2'],
    'max_depth': np.arange(15, 41, 5),      # Profundidad máxima del árbol  
    'min_samples_split': np.arange(6, 21, 2),      # Mínimo de muestras requeridas para dividir un nodo  
    'min_samples_leaf': np.arange(6, 21, 2),        # Mínimo de muestras requeridas para ser una hoja  
}


# Grid para Bernoulli Naive Bayes
param_grid_bnb = {
    'alpha': [0.1, 0.5, 1.0, 10.0],
    'binarize': [0.0, 0.5, 1.0],  # Límite de binarización
    'fit_prior': [True, False]
}



param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200, 500],
    'class_weight': [None, 'balanced']
}

base_ada_param_grid = {
    'n_estimators': [50, 100, 150, 200],  # Número de estimadores  
    'learning_rate': [0.001, 0.01, 0.1, 1],  # Tasa de aprendizaje  
}


param_grid_b_bnb = {  
    **base_ada_param_grid,
    'estimator': [BernoulliNB()],  # Puedes probar otros también  
}
param_grid_b_lr = {  
    **base_ada_param_grid,
    'estimator': [LogisticRegression()],  # Puedes probar otros también  
}

param_grid_fs = {  
    **base_ada_param_grid,
    'estimator': [DecisionTreeClassifier(max_depth=1)]
}

param_grid_rf = {  
    'n_estimators': [250],  # Número de árboles en el bosque  
    'max_depth': [10, 20, 30],  # Profundidad máxima de los árboles  
    'min_samples_split': [7],  # Mínimo de muestras para dividir un nodo  
    'min_samples_leaf': [4],    # Mínimo de muestras en una hoja  
    'max_features': ['log2']  # Características a considerar en la división  
}  
param_grid_svc = {  
    'kernel': ['rbf'],  # Tipos de kernel  
    'gamma': ['auto'],  # Coeficiente del kernel  
}  
