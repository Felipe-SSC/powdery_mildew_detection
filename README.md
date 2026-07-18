# Powdery Mildew Detection

## 1. Descripción del Problema

El mercado de las cerezas es uno en los que Chile tiene mayor participación a nivel mundial, posicionándose en la actualidad como uno de los mayores exportadores de todo el mundo. Por esto mismo, el estudio y seguimiento del correcto crecimiento de las cerezas se ha vuelto clave para asegurar mejores cosechas, aumentar la producción y presentar mayores ingresos para un gran número de entidades, desde agricultores hasta grandes empresas. Es por esto que se postula el diseño de un algoritmo de detección automatizado de la enfermedad 'blanquilla' u 'oídio' en los cultivos de las cerezas, esto a través de modelos de Deep Learning basados en conjuntos de imágenes.

Esta implementación contempla desde un análisis exploratorio de datos (EDA) hasta una implementación inicial y la regularización y comparación de resultados en base a los algoritmos y técnicas probadas.

## 2. Dataset

Para realizar esta tarea, se trabajará con el dataset PlantVillage de 'spMoganty', el cual se puede obtener desde el [repositorio oficial en GitHub](https://github.com/spMohanty/PlantVillage-Dataset.git), o bien desde su página en [Kaggle](https://www.kaggle.com/datasets/mohitsingh1804/plantvillage). Este dataset contiene 38 tipos de plantas, con un total de 54305 imágenes que incluyen plantas sanas y con pestes y 14 especies de cultivos, entre los cuales se encuentran manzanos, naranjos y cerezos.

Para efectos del proyecto, los cultivos relevantes son los cerezos, que dentro del set de datos contemplan 2 carpetas principales, 'train' y 'val', para el entrenamiento y validación respectivamente. En cada uno, se encuentran 2 subcarpetas adicionales, representando los sets de fotos de hojas de cerezas sanas y que presentan la enfermedad estudiada.

En total, el dataset contiene 1906 imágenes, con 1526 y 380 imágenes de entrenamiento y validación, respectivamente, donde se encuentra un promedio de 55.3% de hojas con peste y un 44.7% restante de hojas sanas, presentando así una proporción de 80/20 de training-validation.

## 3. Flujo de trabajo

Para alcanzar resultados, se trabajará primero mediante el preprocesado de imágenes, por medio de procesos de redimensionamiento, transformación y centrado, para poder así dejar las mismas en condiciones que faciliten el trabajo de procesado, así como trabajar en tandas de tamaño 12 para un manejo más amigable en el tiempo de procesamiento. En segundo lugar, la función para medir la pérdida será la entropía cruzada, con un lr de 0.001.

Ya entrando a la etapa de preprocesado, se trabajarán con 5 épocas, promediando los resultados tanto de entrenamiento así como de validación.

---
### Autores a cargo (Nombre de Usuario de GitHub)
- Alejandro Reyes (fleajrv)
- Felipe Sánchez (Felipe-SSC)