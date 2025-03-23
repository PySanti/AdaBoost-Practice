import numpy as np

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

