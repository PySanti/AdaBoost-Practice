import numpy as np

param_grid_rf = {  
    'n_estimators': np.arange(50, 601, 50),
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': np.arange(5, 51, 5),      # Profundidad máxima del árbol  
    'min_samples_split': np.arange(2, 21, 2),      # Mínimo de muestras requeridas para dividir un nodo  
    'min_samples_leaf': np.arange(2, 21, 2),        # Mínimo de muestras requeridas para ser una hoja  
}


# Grid para Bernoulli Naive Bayes
param_grid_bnb = {
    'alpha': [0.1, 0.5, 1.0, 10.0],
    'binarize': [0.0, 0.5, 1.0],  # Límite de binarización
    'fit_prior': [True, False]
}


param_grid_lr = {
    'fit_intercept': [True, False],
    'normalize': [True, False],
    'copy_X': [True, False],  # Para decidir si se debe copiar X o no
    'alpha': [0.1, 1.0, 10.0],  # Solo si usas Ridge o Lasso
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag']
}

