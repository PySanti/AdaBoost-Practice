# AdaBoost

Este proyecto tendra como objetivo implementar todos los conceptos clasicos de un proyecto de machine learning (preprocesamiento, entrenamiento y evaluacion), en combinacion con la tecnica de ensemble learning **Boosting**, en este caso, a traves del algoritmo AdaBoost.

La idea sera estudiar el comportamiento de diferentes algoritmos de clasificacion para diferentes variantes de preprocesamiento de un conjunto de datos y comparar sus resultados con un modelo *boosted*.

El conjunto de datos a utilizar sera un registro de caracteristicas de pacientes de ataques al corazon.

Los algoritmos a utilizar seran:

* Regresion Lineal
* Naive Bayes
* Random Forest

## Preprocesamiento

Shape del dataset : 319795 x 18 

1- Manejo de Nans: no cuenta con valores Nan.

2- Manejo de variables categoricas: la lista de columnas categoricas con su cantidad de categoricas es la siguiente.

Smoking 2\
AlcoholDrinking 2\
Stroke 2\
DiffWalking 2\
Sex 2\
AgeCategory 13\
Race 6\
Diabetic 4\
PhysicalActivity 2\
GenHealth 5\
Asthma 2\
KidneyDisease 2\
SkinCancer 2

Son 13 columnas categoricas y la cantidad de categorias (y por lo tanto de columnas nuevas) es 46.

Por lo tanto, si utilizasemos OneHotEncoding, las columnas aumentarian de 18 a 52. Un numero bastante razonable.


3- Estudio de desequilibrio de datos: no se realizara.

4- Estudio de correlaciones: se reemplazara por un estudio de extraccion y/o seleccion de caracteristicas.

5- Estudio de distribucion gaussiana de los datos: no se realizara.

6- Extraccion y/o seleccion de caracteristicas: se incluyo en el pipeline de manera intrinseca la seleccion de caracteristicas utilizando el transformador `BestFeatures` y se agrego la posibilidad de generar el dataset con `PCA`.

Despues de ejecutar el proceso de seleccion de caracteristicas, se redujo la dimensionalidad a 23 caracteristicas.

Despues de ejecutar PCA, se redujo la dimensionalidad a 17 caracteristicas. 

7- Estudio de casos anomalos: se utilizara solo sobre el conjunto de entrenamiento.

Despues de ejecutar el proceso de estudio de casos anomalos sobre el conjunto de entrenamieto utilizando el algoritmo `Isolation Forest` con 400 arboles, se detectaron 16% de casos anomalos.



8- Scalers.

## Entrenamiento
