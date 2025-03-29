# AdaBoost

Este proyecto tendra como objetivo implementar todos los conceptos clasicos de un proyecto de machine learning (preprocesamiento, entrenamiento y evaluacion), en combinacion con la tecnica de ensemble learning **Boosting**, en este caso, a traves del algoritmo AdaBoost.

La idea sera estudiar el comportamiento de diferentes algoritmos de clasificacion para diferentes variantes de preprocesamiento de un conjunto de datos y comparar sus resultados con un modelo *boosted*. Es decir:

```
Variante de preprocesamiento 1:

Naive Bayes : x
Naive Bayes Boosted : y
...
```

El conjunto de datos a utilizar sera un registro de caracteristicas de pacientes de ataques al corazon.

Los algoritmos a utilizar seran:

* Regresion Lineal
* Naive Bayes
* Random Forest

## Preprocesamiento

```
Shape del dataset

319795 x 18

Distribucion del target

No : 91.4%
Yes: 8.6%
```

1- Manejo de Nans: no cuenta con valores Nan.

2- **Manejo de variables categoricas**: la lista de columnas categoricas con su cantidad de categoricas es la siguiente.

```
Smoking 2
AlcoholDrinking 2
Stroke 2
DiffWalking 2
Sex 2
AgeCategory 13
Race 6
Diabetic 4
PhysicalActivity 2
GenHealth 5
Asthma 2
KidneyDisease 2
SkinCancer 2
```
Son 13 columnas categoricas y la cantidad de categorias (y por lo tanto de columnas nuevas) es 46.

Por lo tanto, si utilizasemos OneHotEncoding, las columnas aumentarian de 18 a 52. Un numero bastante razonable.


3- Estudio de desequilibrio de datos: no se realizara.

4- Estudio de correlaciones: se reemplazara por un estudio de extraccion y/o seleccion de caracteristicas.

5- Estudio de distribucion gaussiana de los datos: no se realizara.

6- **Extraccion y seleccion de caracteristicas**: se incluyo en el pipeline la seleccion de caracteristicas utilizando el transformador `BestFeatures` y se agrego la posibilidad de generar el dataset con `PCA`.

Despues de ejecutar el proceso de seleccion de caracteristicas, se redujo la dimensionalidad a 23 caracteristicas.

Despues de ejecutar PCA, se redujo la dimensionalidad a 17 caracteristicas. 

7- **Estudio de casos anomalos**: se utilizara solo sobre el conjunto de entrenamiento.

Despues de ejecutar el proceso de estudio de casos anomalos sobre el conjunto de entrenamieto utilizando el algoritmo `Isolation Forest` con 400 arboles, se detectaron 16% de casos anomalos.



8- **Scalers**: se incluiran como una alternativa de preprocesamiento.

## Entrenamiento


Es importante destacar que, al final del dia este es un proyecto con fines didacticos y no tiene todo el sentido que deberia tener. Esto es especifica basicamente por que no es tan comun utilizar AdaBoost con algoritmos como *Naive Bayes* o *Regresion Logistica*. Esto es por que, AdaBoost implementa el concepto de *Boosting* el cual en la practica es similar al concepto del bagging en el sentido de que no tiene sentido implementarlos si los modelos que se utilizan de base no son "debiles".

AdaBoost se suele implementar con arboles de decision por que son modelos "debiles" por si solos, sin embargo, *Naive Bayes* o *Regresion Logistica* no lo son.

Independientemente del caso, aca los resultados para cada variante de preprocesamiento:

Importante: destacar que las precisiones son el *f1_score* para la clase positiva (minoritaria).


### Naive Bayes

| Variante de preprocesamiento 	|                                                                            Naive Bayes                                                                           	| Boosted Naive Bayes                                                           	|
|:----------------------------:	|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:	|-------------------------------------------------------------------------------	|
|  + Scaler; + PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'alpha': 0.1, 'binarize': 0.0, 'fit_prior': False}<br>    Performance en train : 0.23<br>    Performance en test : 0.24  	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  + Scaler; - PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'alpha': 10.0, 'binarize': 0.0, 'fit_prior': False}<br>    Performance en train : 0.23<br>    Performance en test : 0.30 	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  - Scaler; + PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'alpha': 0.1, 'binarize': 1.0, 'fit_prior': False}<br>    Performance en train : 0.18<br>    Performance en test : 0.19  	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  - Scaler; - PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'alpha': 10.0, 'binarize': 0.0, 'fit_prior': False}<br>    Performance en train : 0.23<br>    Performance en test : 0.30 	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.01 	|
| + Scaler; + PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'alpha': 0.1, 'binarize': 0.0, 'fit_prior': False}<br>    Performance en train : 0.25<br>    Performance en test : 0.24  	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
| + Scaler; - PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'alpha': 1.0, 'binarize': 0.5, 'fit_prior': True}<br>    Performance en train : 0.33<br>    Performance en test : 0.33   	| Performance boosted en train : 0.10<br>    Performance boosted en test : 0.11 	|
| - Scaler; + PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'alpha': 0.1, 'binarize': 1.0, 'fit_prior': False}<br>    Performance en train : 0.21<br>    Performance en test : 0.22  	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
| - Scaler; - PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'alpha': 1.0, 'binarize': 0.0, 'fit_prior': True}<br>    Performance en train : 0.34<br>    Performance en test : 0.34   	| Performance boosted en train : 0.07<br>    Performance boosted en test : 0.07 	|



### Regresion Logistica

| Variante de preprocesamiento 	|                                                                                              Regresion Logistica                                                                                              	| Boosted Regresion Logistica                                                   	|
|:----------------------------:	|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:	|-------------------------------------------------------------------------------	|
|  + Scaler; + PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'C': 0.01, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}<br>    Performance en train : 0.28<br>    Performance en test : 0.30  	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  + Scaler; - PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'saga'}<br>    Performance en train : 0.24<br>    Performance en test : 0.29      	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  - Scaler; + PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l1', 'solver': 'saga'}<br>    Performance en train : 0.20<br>    Performance en test : 0.22      	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
|  - Scaler; - PCA; + Outliers 	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga'}<br>    Performance en train : 0.23<br>    Performance en test : 0.29      	| Performance boosted en train : 0.02<br>    Performance boosted en test : 0.05 	|
| + Scaler; + PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'C': 0.01, 'class_weight': 'balanced', 'max_iter': 200, 'penalty': 'l1', 'solver': 'saga'}<br>    Performance en train : 0.30<br>    Performance en test : 0.30       	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
| + Scaler; - PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 500, 'penalty': 'l2', 'solver': 'saga'}<br>    Performance en train : 0.34<br>    Performance en test : 0.34      	| Performance boosted en train : 0.17<br>    Performance boosted en test : 0.17 	|
| - Scaler; + PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 500, 'penalty': 'l1', 'solver': 'saga'}<br>    Performance en train : 0.23<br>    Performance en test : 0.23      	| Performance boosted en train : 0.00<br>    Performance boosted en test : 0.00 	|
| - Scaler; - PCA; - Outliers  	| Mejor combinacion de hiperparametros : {'C': 0.001, 'class_weight': 'balanced', 'max_iter': 100, 'penalty': 'l2', 'solver': 'liblinear'}<br>    Performance en train : 0.33<br>    Performance en test : 0.33 	| Performance boosted en train : 0.16<br>    Performance boosted en test : 0.17 	|



Cómo vemos, de por sí ninguno de los algoritmos por separado logró hacer buenas predicciones, a qué se debe esto? Esta situación puede estar siendo causada por varias razones:

1- Los algoritmos, al construir límites de decisión lineales, no son capaces de capturar los patrones entre los datos.

2- Está habiendo algún error importante en el proceso de preprocesamiento (menos probable).

3- El hecho de que el conjunto de datos esté desequilibrado afecta negativamente a estos algoritmos concretos. (muy probable)

En un caso de uso real, lo primero que haría sería implementar SVC y MLP para comprobar cuál de las opciones anteriores es la correcta. Además de eso, tratar de utilizar *ensemble learning* para mejorar las predicciones sin alterar artificialmente el conjunto de datos utilizando técnicas como SMOTE.

Ademas de lo anterior, AdaBoost esta generando resultados inesperadamente bajos. La razon de esto seguramente sea que los algoritmos Naive Bayes y Regresion Logistica no son los mas apropiados para utilizar con AdaBoost al no ser lo "suficientemente debiles".


Para realizar un estudio mas completo, se haran las siguientes acciones:

1- Volver a estudiar los algoritmos anteriores eliminando variantes de preprocesamiento inutiles (PCA y Outliers) y reemplazar por tenicas de manejo de desbalance (SMOTE o ADASYN).

2- Por cada algoritmo, se realizara un proceso de seleccion de modelo para su version Boosted tambien.

3- Aparte, se generara otra matriz de estudio para analizar el comportamiento de Stump Forest, un algoritmo de AdaBoost con Arboles de decision (su implementacion mas comun), esto para comparar los resultados anteriores.


Después de realizar las acciones anteriores, concluimos lo siguiente:

Con respecto a la implementación de SMOTE y ADASYN, ambas lograron resultados similares, mejorando mucho el rendimiento de los modelos para el conjunto de entrenamiento, pero no mejorándolo para nada para el conjunto de test.

Esto nos llevó a pensar que la única alternativa que podría estar causando problemas de precisión es que simplemente los algoritmos que estamos utilizando (Naive Bayes y Regresión Logística) al construir límites de decisión lineales, no están siendo capaces de capturar patrones complejos en el dataset.

La teoria anterior fue refutada cuando probamos utilizando SVC, el cual no dio resultados mejores aun utilizando kernel polinomico. 

Por último, el algoritmo forest stumps funcionó bastante bien, en algunos casos hasta mejor que Naive Bayes y Regresion Logistica. Acá los resultados:


#### Resultados base

|  Preprocesamiento 	|                               NB                              	|                               LR                              	|                              SVC                              	|
|:-----------------:	|:-------------------------------------------------------------:	|:-------------------------------------------------------------:	|:-------------------------------------------------------------:	|
| + Scaler; + SMOTE 	| Performance en train : 0.72<br>    Performance en test : 0.33 	| Performance en train : 0.74<br>    Performance en test : 0.33 	| Performance en train : 0.76<br>    Performance en test : 0.31 	|
| + Scaler; - SMOTE 	| Performance en train : 0.33<br>    Performance en test : 0.33 	| Performance en train : 0.34<br>    Performance en test : 0.34 	|                               -                               	|
| - Scaler; + SMOTE 	| Performance en train : 0.72<br>    Performance en test : 0.33 	| Performance en train : 0.74<br>    Performance en test : 0.33 	|                               -                               	|
| - Scaler; - SMOTE 	| Performance en train : 0.33<br>    Performance en test : 0.33 	| Performance en train : 0.34<br>    Performance en test : 0.34 	|                               -                               	|

#### Resultados Boosted

|  Preprocesamiento 	|                           Boosted NB                          	|                           Boosted LR                          	|                               FS                              	|
|:-----------------:	|:-------------------------------------------------------------:	|:-------------------------------------------------------------:	|:-------------------------------------------------------------:	|
| + Scaler; + SMOTE 	| Performance en train : 0.83<br>    Performance en test : 0.33 	| Performance en train : 0.73<br>    Performance en test : 0.30 	| Performance en train : 0.86<br>    Performance en test : 0.34 	|
| + Scaler; - SMOTE 	| Performance en train : 0.12<br>    Performance en test : 0.12 	| Performance en train : 0.16<br>    Performance en test : 0.16 	| Performance en train : 0.16<br>    Performance en test : 0.16 	|
| - Scaler; + SMOTE 	| Performance en train : 0.82<br>    Performance en test : 0.33 	| Performance en train : 0.74<br>    Performance en test : 0.31 	| Performance en train : 0.86<br>    Performance en test : 0.34 	|
| - Scaler; - SMOTE 	| Performance en train : 0.12<br>    Performance en test : 0.12 	| Performance en train : 0.17<br>    Performance en test : 0.17 	| Performance en train : 0.16<br>    Performance en test : 0.16 	|


Dejaremos este proyecto en este punto y mas adelante lo retomaremos para tratar de averiguar la razon de estos problemas.