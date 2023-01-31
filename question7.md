### Accuracy

Es el número de predicciones correctas dividido por el número total de predicciones. 

En clasificación multietiqueta accuracy se calcula para el subconjunto, y calcula la fracción de veces que todas las etiquetas predichas para una muestra coinciden exactamente con las etiquetas reales.


### Hamming Loss

Es una métrica de evaluación basada en instancias que evalúa cuántas veces un par instancia-etiqueta se clasifica mal. Hamming Loss penaliza las etiquetas individuales que fueron clasificadas incorrectamente. No tiene en cuenta la relación entre etiquetas correctas e incorrectas.

### Precision

Se refiere a la fracción de resultados verdaderos positivos de todos los resultados positivos. Mide la estabilidad de la medida frente a las repeticiones.

De todas las predicciones positivas, ¿cuántas son realmente positivas?

### Recall

Calcula la fracción de resultados verdaderos positivos que fueron correctamente identificados. 

De todos los casos positivos reales, ¿cuántos son positivos predichos?

### F1_Score

El puntaje F1 es la media armónica entre la precisión y el recall. Su objetivo sería maximizar tanto la precisión como la recuperación.

En una tarea de clasificación multiclase y multietiqueta, se pueden aplicar las métricas de precisión, recall y F-measures a cada etiqueta de forma independiente. Hay varias maneras de combinar los resultados a través de las etiquetas, especificadas por el argumento de promedio en las funciones average_precision_score (sólo multietiqueta), f1_score, fbeta_score, precision_recall_fscore_support, precision_score y recall_score.