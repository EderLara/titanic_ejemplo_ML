# Documentación: Análisis de Supervivencia del Titanic

## Descripción General

El archivo `ejemplo_titanic.py` implementa un flujo de trabajo completo de machine learning supervisado para predecir qué pasajeros sobrevivieron al naufragio del Titanic. Es un ejemplo práctico que cubre todo el ciclo de vida de un proyecto de ciencia de datos.

**Autores:** Tania Rodriguez - Eder Lara  
**Fecha:** 6 de junio de 2025

## Arquitectura del Sistema

### Bibliotecas Utilizadas

#### Análisis y Manipulación de Datos
- `pandas`: Manipulación y análisis de datos estructurados
- `numpy`: Operaciones numéricas y arrays multidimensionales

#### Visualización
- `matplotlib.pyplot`: Creación de gráficos y visualizaciones
- `seaborn`: Visualizaciones estadísticas avanzadas

#### Machine Learning (Scikit-learn)
- **Preprocesamiento:**
  - `train_test_split`: División de datos en entrenamiento y prueba
  - `StandardScaler`: Normalización de características numéricas
  - `OneHotEncoder`: Codificación de variables categóricas
  - `SimpleImputer`: Imputación de valores faltantes
  - `ColumnTransformer`: Aplicación de transformaciones por tipo de columna
  - `Pipeline`: Creación de flujos de preprocesamiento

- **Modelos:**
  - `LogisticRegression`: Regresión logística
  - `DecisionTreeClassifier`: Árbol de decisión
  - `RandomForestClassifier`: Bosque aleatorio
  - `SVC`: Máquinas de vectores de soporte

- **Evaluación:**
  - `cross_val_score`: Validación cruzada
  - `GridSearchCV`: Búsqueda de hiperparámetros
  - Métricas: accuracy, precision, recall, f1-score, ROC-AUC
  - `confusion_matrix`: Matriz de confusión
  - `classification_report`: Reporte detallado de clasificación

## Funciones Implementadas

### 1. `cargar_datos()`
**Propósito:** Carga los datos del Titanic desde archivo local o descarga desde GitHub.

**Funcionalidad:**
- Intenta cargar desde archivo local `titanic.csv`
- Si no existe, descarga desde repositorio público
- Guarda los datos localmente para uso futuro

**Retorna:** DataFrame de pandas con los datos del Titanic

### 2. `explorar_datos(data)`
**Propósito:** Realiza análisis exploratorio de datos (EDA).

**Funcionalidad:**
- Muestra información básica del dataset (primeras filas, info, estadísticas)
- Identifica valores faltantes
- Crea y guarda visualizaciones:
  - Distribución de supervivencia
  - Tasa de supervivencia por sexo
  - Tasa de supervivencia por clase
  - Distribución de edades por supervivencia
  - Tasa de supervivencia por tamaño de familia
- Crea nueva característica: `FamilySize` (SibSp + Parch + 1)

**Archivos generados:**
- `titanic_supervivencia.png`
- `titanic_supervivencia_sexo.png`
- `titanic_supervivencia_clase.png`
- `titanic_edad_supervivencia.png`
- `titanic_familia_supervivencia.png`

### 3. `preparar_datos(data)`
**Propósito:** Prepara los datos para el modelado.

**Funcionalidad:**
- Selecciona características relevantes: `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']`
- Divide datos en entrenamiento (80%) y prueba (20%) con estratificación
- Crea pipelines de preprocesamiento:
  - **Numéricas:** Imputación con mediana + normalización
  - **Categóricas:** Imputación con moda + one-hot encoding
- Utiliza `ColumnTransformer` para aplicar transformaciones específicas

**Retorna:** X_train, X_test, y_train, y_test, preprocessor

### 4. `entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor)`
**Propósito:** Entrena múltiples modelos y evalúa su rendimiento.

**Modelos evaluados:**
- Regresión Logística
- Árbol de Decisión
- Random Forest
- SVM (Support Vector Machine)

**Proceso de evaluación:**
- Validación cruzada de 5 folds
- Selección del mejor modelo basado en accuracy
- Evaluación en conjunto de prueba
- Generación de métricas completas
- Visualizaciones de rendimiento

**Archivos generados:**
- `titanic_comparacion_modelos.png`
- `titanic_matriz_confusion.png`
- `titanic_curva_roc.png`

**Retorna:** Mejor pipeline, nombre del mejor modelo, resultados de todos los modelos

### 5. `optimizar_hiperparametros(best_pipeline, best_model_name, X_train, y_train, X_test, y_test)`
**Propósito:** Optimiza hiperparámetros del mejor modelo usando Grid Search.

**Espacios de búsqueda por modelo:**
- **Regresión Logística:** C, solver, penalty
- **Árbol de Decisión:** max_depth, min_samples_split, min_samples_leaf
- **Random Forest:** n_estimators, max_depth, min_samples_split, min_samples_leaf
- **SVM:** C, gamma, kernel

**Funcionalidad:**
- Grid Search con validación cruzada de 5 folds
- Evaluación del modelo optimizado
- Comparación de métricas antes y después de la optimización

**Retorna:** Modelo optimizado

### 6. `interpretar_modelo(model, best_model_name, X_test, y_test, preprocessor)`
**Propósito:** Interpreta el modelo final para entender la importancia de características.

**Funcionalidad:**
- **Para árboles/Random Forest:** Extrae importancia de características
- **Para Regresión Logística:** Analiza coeficientes
- Análisis de errores por características demográficas
- Identificación de patrones en predicciones incorrectas

**Archivos generados:**
- `titanic_importancia_caracteristicas.png` o `titanic_coeficientes.png`
- `titanic_errores_sexo.png`
- `titanic_errores_clase.png`
- `titanic_errores_edad.png`

### 7. `hacer_prediccion(model, new_data)`
**Propósito:** Realiza predicciones con nuevos datos.

**Funcionalidad:**
- Predice clase (sobrevive/no sobrevive)
- Calcula probabilidad de supervivencia
- Retorna resultados en formato DataFrame

### 8. `main()`
**Propósito:** Función principal que orquesta todo el flujo de trabajo.

**Flujo completo:**
1. Carga de datos
2. Análisis exploratorio
3. Preparación de datos
4. Entrenamiento y evaluación de modelos
5. Optimización de hiperparámetros
6. Interpretación del modelo
7. Ejemplo de predicción con nuevos datos

## Características del Dataset

### Variables de Entrada (Features)
- **Pclass:** Clase del pasajero (1, 2, 3)
- **Sex:** Sexo del pasajero (male, female)
- **Age:** Edad del pasajero
- **SibSp:** Número de hermanos/cónyuges a bordo
- **Parch:** Número de padres/hijos a bordo
- **Fare:** Tarifa pagada
- **Embarked:** Puerto de embarque (C, Q, S)

### Variable Objetivo (Target)
- **Survived:** Supervivencia (0 = No, 1 = Sí)

### Características Derivadas
- **FamilySize:** Tamaño de familia (SibSp + Parch + 1)

## Métricas de Evaluación

El sistema evalúa los modelos usando múltiples métricas:

- **Accuracy:** Proporción de predicciones correctas
- **Precision:** Proporción de verdaderos positivos entre predicciones positivas
- **Recall:** Proporción de verdaderos positivos identificados
- **F1-Score:** Media armónica entre precision y recall
- **ROC-AUC:** Área bajo la curva ROC

## Configuración y Estilo

### Visualizaciones
- Estilo: `seaborn-v0_8-whitegrid`
- Tamaño de figura: 12x8 pulgadas
- Tamaño de fuente: 12
- Resolución: 300 DPI
- Formato: PNG con bbox_inches='tight'

### Reproducibilidad
- Semilla aleatoria: 42 (para todos los componentes aleatorios)
- Estratificación en división de datos
- Configuración determinística de modelos

## Archivos de Salida

El script genera múltiples archivos de visualización:
1. `titanic_supervivencia.png` - Distribución general
2. `titanic_supervivencia_sexo.png` - Análisis por género
3. `titanic_supervivencia_clase.png` - Análisis por clase social
4. `titanic_edad_supervivencia.png` - Análisis por edad
5. `titanic_familia_supervivencia.png` - Análisis por tamaño de familia
6. `titanic_comparacion_modelos.png` - Comparación de algoritmos
7. `titanic_matriz_confusion.png` - Matriz de confusión
8. `titanic_curva_roc.png` - Curva ROC
9. `titanic_importancia_caracteristicas.png` - Importancia de variables
10. `titanic_errores_*.png` - Análisis de errores

## Casos de Uso

Este script es ideal para:
- **Educación:** Aprender conceptos de machine learning
- **Prototipado:** Base para proyectos de clasificación binaria
- **Benchmarking:** Comparar diferentes algoritmos
- **Análisis exploratorio:** Entender patrones en datos históricos

## Extensibilidad

El código está diseñado de manera modular, permitiendo:
- Agregar nuevos modelos fácilmente
- Modificar espacios de búsqueda de hiperparámetros
- Incorporar nuevas métricas de evaluación
- Extender el análisis exploratorio
- Personalizar visualizaciones

## Consideraciones Técnicas

### Manejo de Datos Faltantes
- **Numéricas:** Imputación con mediana
- **Categóricas:** Imputación con moda

### Preprocesamiento
- Normalización de variables numéricas (StandardScaler)
- Codificación one-hot para categóricas
- Pipeline automatizado para reproducibilidad

### Validación
- Validación cruzada de 5 folds
- División estratificada para mantener proporción de clases
- Evaluación en conjunto de prueba independiente