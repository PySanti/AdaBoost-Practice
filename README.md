# AdaBoost

Este proyecto tendrá como objetivo implementar todos los conceptos clásicos de un proyecto de machine learning (preprocesamiento, entrenamiento y evaluación), en combinación con la técnica de ensemble learning **Boosting**, en este caso, a través del algoritmo AdaBoost.

La idea será estudiar el comportamiento de diferentes algoritmos de clasificación para diferentes variantes de preprocesamiento de un conjunto de datos y comparar sus resultados con un modelo *boosted*. Es decir:

```
Variante de preprocesamiento 1:

Naive Bayes : x
Naive Bayes Boosted : y
...
```

El conjunto de datos a utilizar será un registro de características de pacientes de ataques al corazón.

Los algoritmos a utilizar serán:

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

2- **Manejo de variables categóricas**: la lista de columnas categóricas con su cantidad de categorías es la siguiente.

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
Son 13 columnas temáticas y la cantidad de categorías (y, por lo tanto, de columnas nuevas) es 46.

Por lo tanto, si utilizásemos OneHotEncoding, las columnas aumentarían de 18 a 52. Un número bastante razonable.


3- Estudio de desequilibrio de datos: no se realizará.

4- Estudio de correlaciones: se reemplazará por un estudio de extracción y/o selección de características.

5- Estudio de distribución gaussiana de los datos: no se realizará.

6- **Extracción y selección de características**: se incluyó en el pipeline la selección de características utilizando el transformador `BestFeatures` y se agregó la posibilidad de generar el dataset con `PCA`.

Después de ejecutar el proceso de selección de características, se redujo la dimensionalidad a 23 características.

Después de ejecutar PCA, se redujo la dimensionalidad a 17 características. 

7- **Estudio de casos anómalos**: se utilizará solo sobre el conjunto de entrenamiento.

Después de ejecutar el proceso de estudio de casos anómalos sobre el conjunto de entrenamiento utilizando el algoritmo `Isolation Forest` con 400 árboles, se detectaron 16% de casos anómalos.



8- **Scalers**: se incluirán como una alternativa de preprocesamiento.

## Entrenamiento
