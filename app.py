import streamlit as st
import pandas as pd
from io import StringIO
import ejemplo_titanic
import sys

# --- Importar funciones desde tu script original ---
# Asegúrate de que 'ejemplo_titanic.py' esté en la misma carpeta
try:
    from ejemplo_titanic import (
        cargar_datos,
        explorar_datos,
        preparar_datos,
        entrenar_evaluar_modelos,
        interpretar_modelo,
        hacer_prediccion
    )
except ImportError:
    st.error("Error: Asegúrate de que el archivo 'ejemplo_titanic.py' se encuentra en la misma carpeta que 'app.py'.")
    st.stop()

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Supervivencia del Titanic",
    page_icon="🚢",
    layout="wide"
)

st.title("🚢 Análisis de Supervivencia del Titanic")
st.write("Esta aplicación interactiva te permite seguir un flujo de trabajo de Machine Learning paso a paso, desde la carga de datos hasta la predicción.")

# --- Inicialización del estado de la sesión ---
# El estado de la sesión se usa para guardar variables entre interacciones
if 'data' not in st.session_state:
    st.session_state.data = None
if 'prepared_data' not in st.session_state:
    st.session_state.prepared_data = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'model_name' not in st.session_state:
    st.session_state.model_name = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None

# --- Creación de Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Carga de Datos",
    "2. Exploración",
    "3. Preparación",
    "4. Entrenamiento",
    "5. Interpretación",
    "6. Predicción"
])

# --- Pestaña 1: Carga de Datos ---
with tab1:
    st.header("Paso 1: Cargar el Conjunto de Datos")
    st.info("Haz clic en el botón para cargar los datos del Titanic. Si no existe el archivo local, se descargará automáticamente.")

    if st.button("Cargar Datos"):
        with st.spinner("Cargando datos..."):
            st.session_state.data = cargar_datos()
        st.success("¡Datos cargados exitosamente!")
        st.dataframe(st.session_state.data.head())


# --- Pestaña 2: Exploración de Datos ---
with tab2:
    st.header("Paso 2: Análisis Exploratorio de Datos (EDA)")
    if st.session_state.data is not None:
        if st.button("Explorar Datos"):
            with st.spinner("Generando visualizaciones..."):
                # Capturamos los prints para mostrarlos en la app
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                # Ejecutamos la función y guardamos las figuras
                df_explorado = explorar_datos(st.session_state.data.copy())
                
                # Restauramos la salida estándar
                sys.stdout = old_stdout
                
                st.subheader("Información y Estadísticas")
                st.text(captured_output.getvalue())

                st.subheader("Visualizaciones")
                # Mostramos las imágenes guardadas por la función
                st.image('titanic_supervivencia.png', caption='Distribución de Supervivencia')
                st.image('titanic_supervivencia_sexo.png', caption='Tasa de Supervivencia por Sexo')
                st.image('titanic_supervivencia_clase.png', caption='Tasa de Supervivencia por Clase')
                st.image('titanic_edad_supervivencia.png', caption='Distribución de Edades por Supervivencia')
                st.image('titanic_familia_supervivencia.png', caption='Tasa de Supervivencia por Tamaño de Familia')
                
                # Guardamos el dataframe con la nueva columna 'FamilySize'
                st.session_state.data = df_explorado
                st.success("Análisis exploratorio completado.")
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

# --- Pestaña 3: Preparación de Datos ---
with tab3:
    st.header("Paso 3: Preparar los Datos para el Modelo")
    if st.session_state.data is not None:
        if st.button("Preparar Datos"):
            with st.spinner("Dividiendo y preprocesando los datos..."):
                X_train, X_test, y_train, y_test, preprocessor = preparar_datos(st.session_state.data)
                
                # Guardamos los resultados en el estado de la sesión
                st.session_state.prepared_data = (X_train, X_test, y_train, y_test)
                st.session_state.preprocessor = preprocessor
                
                st.success("Datos preparados exitosamente.")
                st.info(f"Tamaño del conjunto de entrenamiento: {X_train.shape[0]} muestras")
                st.info(f"Tamaño del conjunto de prueba: {X_test.shape[0]} muestras")
                st.write("Vista previa de los datos de entrenamiento (X_train):")
                st.dataframe(X_train.head())
    else:
        st.warning("Por favor, carga los datos en la Pestaña 1 (Cargar Datos) primero.")

# --- Pestaña 4: Entrenamiento y Evaluación ---
with tab4:
    st.header("Paso 4: Entrenar y Evaluar Múltiples Modelos")
    if st.session_state.prepared_data is not None:
        if st.button("Entrenar y Evaluar Modelos"):
            with st.spinner("Entrenando modelos y evaluando... Esto puede tardar un momento."):
                X_train, X_test, y_train, y_test = st.session_state.prepared_data
                preprocessor = st.session_state.preprocessor
                
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                best_pipeline, best_model_name, results = entrenar_evaluar_modelos(X_train, X_test, y_train, y_test, preprocessor)

                sys.stdout = old_stdout
                
                st.subheader("Resultados de la Validación Cruzada")
                st.text(captured_output.getvalue().split('Rendimiento en el conjunto de prueba:')[0])
                st.image('titanic_comparacion_modelos.png', caption='Comparación de Modelos')

                st.subheader(f"Rendimiento del Mejor Modelo ({best_model_name}) en el Conjunto de Prueba")
                st.text('Rendimiento en el conjunto de prueba:' + captured_output.getvalue().split('Rendimiento en el conjunto de prueba:')[1])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image('titanic_matriz_confusion.png', caption='Matriz de Confusión')
                with col2:
                    st.image('titanic_curva_roc.png', caption='Curva ROC')

                # Guardamos el mejor modelo
                st.session_state.best_model = best_pipeline
                st.session_state.model_name = best_model_name
                st.success(f"Entrenamiento completado. El mejor modelo es: **{best_model_name}**")
    else:
        st.warning("Por favor, prepara los datos en la Pestaña 3 (Preparar los Datos) primero.")


# --- Pestaña 5: Interpretación del Modelo ---
with tab5:
    st.header("Paso 5: Interpretar el Mejor Modelo")
    if st.session_state.best_model is not None:
        if st.button("Interpretar Mejor Modelo"):
            with st.spinner("Generando análisis de interpretabilidad..."):
                X_train, X_test, y_train, y_test = st.session_state.prepared_data
                
                old_stdout = sys.stdout
                sys.stdout = captured_output = StringIO()

                interpretar_modelo(
                    st.session_state.best_model,
                    st.session_state.model_name,
                    X_test, y_test,
                    st.session_state.preprocessor
                )

                sys.stdout = old_stdout
                
                st.subheader("Importancia de las Características")
                if st.session_state.model_name in ['Random Forest', 'Árbol de Decisión']:
                    st.image('titanic_importancia_caracteristicas.png')
                elif st.session_state.model_name == 'Regresión Logística':
                    st.image('titanic_coeficientes.png')
                else:
                    st.info("La interpretabilidad directa de características no está implementada para SVM en este script.")

                st.subheader("Análisis de Errores")
                st.text(captured_output.getvalue())
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image('titanic_errores_sexo.png')
                with col2:
                    st.image('titanic_errores_clase.png')
                with col3:
                    st.image('titanic_errores_edad.png')
    else:
        st.warning("Por favor, entrena un modelo en la Pestaña 4 (Entrenar y Evaluar Múltiples Modeloso) primero.")


# --- Pestaña 6: Predicción ---
with tab6:
    st.header("Paso 6: Realizar una Predicción con Nuevos Datos")
    if st.session_state.best_model is not None:
        st.info("Completa el siguiente formulario para predecir la supervivencia de un nuevo pasajero.")
        
        with st.form("prediction_form"):
            # Creamos columnas para un mejor diseño
            col1, col2, col3 = st.columns(3)
            
            with col1:
                pclass = st.selectbox("Clase (Pclass)", [1, 2, 3])
                sex = st.selectbox("Sexo (Sex)", ["male", "female"])
                embarked = st.selectbox("Puerto de Embarque (Embarked)", ["C", "Q", "S"])
            
            with col2:
                age = st.number_input("Edad (Age)", min_value=0, max_value=100, value=30)
                sibsp = st.number_input("Nº de Hermanos/Cónyuges (SibSp)", min_value=0, max_value=10, value=0)
            
            with col3:
                parch = st.number_input("Nº de Padres/Hijos (Parch)", min_value=0, max_value=10, value=0)
                fare = st.number_input("Tarifa (Fare)", min_value=0.0, value=50.0, step=0.1, format="%.2f")

            submit_button = st.form_submit_button(label="Predecir Supervivencia")

        if submit_button:
            # Crear el DataFrame para el nuevo pasajero
            new_passenger = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [sex],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Embarked': [embarked]
            })
            
            st.write("Datos del nuevo pasajero:")
            st.dataframe(new_passenger)
            
            # Realizar la predicción
            with st.spinner("Realizando predicción..."):
                prediction_result = hacer_prediccion(st.session_state.best_model, new_passenger)
                
                pred = prediction_result['Predicción'].iloc[0]
                prob = prediction_result['Probabilidad de Supervivencia'].iloc[0]

            st.subheader("Resultado de la Predicción")
            if pred == 1:
                st.success(f"**El pasajero SOBREVIVIRÍA** con una probabilidad del {prob:.2%}")
                st.balloons()
            else:
                st.error(f"**El pasajero NO SOBREVIVIRÍA** (Probabilidad de supervivencia: {prob:.2%})")
                st.snow()

    else:
        st.warning("Por favor, completa todos los pasos anteriores para poder hacer una predicción.")