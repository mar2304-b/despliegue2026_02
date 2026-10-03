import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("Predicción de Aprobación de Curso")
st.write("Esta aplicación procesa las variables de entrada y realiza predicciones utilizando un modelo de Bagging pre-entrenado.")

# Crear pestañas para las dos opciones de entrada de datos
tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Archivo (Carga Masiva)"])

with tab1:
    st.header("Datos de Entrada Individual")
    
    # Opciones para la variable Felder obtenidas de los datos reales o previstos
    opciones_felder = ['sensorial', 'activo', 'visual', 'equilibrio', 'secuencial', 'reflexivo', 'verbal', 'intuitivo']
    
    # Entradas del formulario
    felder_input = st.selectbox("Selecciona el estilo de aprendizaje (Felder):", opciones_felder, key="felder_individual")
    examen_input = st.number_input("Examen de Admisión:", min_value=0.0, max_value=5.0, value=3.83, step=0.01, key="examen_individual")
    
    # Crear un DataFrame temporal con los datos del usuario
    data_dict = {
        'Felder': [felder_input],
        'Examen_admisión': [examen_input]
    }
    df_input = pd.DataFrame(data_dict)
    
    if st.button("Realizar Predicción Individual", key="btn_individual"):
        try:
            # Copia de trabajo
            df_procesado = df_input.copy()
            
            # 2. Cargar y aplicar el transformador/columnas de One-Hot para la variable Felder
            one_hot_transformer = joblib.load('one_hot_columns.joblib')
            
            if isinstance(one_hot_transformer, list):
                si_columnas_one_hot = [col for col in one_hot_transformer if 'Felder_' in col]
                for col_name in si_columnas_one_hot:
                    valor_esperado = col_name.replace('Felder_', '')
                    df_procesado[col_name] = (df_procesado['Felder'] == valor_esperado).astype(int)
            else:
                df_encoded = pd.get_dummies(df_procesado[['Felder']])
                df_procesado = pd.concat([df_procesado, df_encoded], axis=1)
                si_columnas_one_hot = [col for col in df_procesado.columns if 'Felder_' in col]
            
            # Eliminar la variable original Felder
            df_procesado = df_procesado.drop(columns=['Felder'], errors='ignore')
            
            # Asegurar que existan todas las columnas que el modelo espera
            if isinstance(one_hot_transformer, list):
                for col in si_columnas_one_hot:
                    if col not in df_procesado.columns:
                        df_procesado[col] = 0
            
            # 3. Normalizar la variable Examen_admisión con 'min_max_scaler.joblib'
            scaler = joblib.load('min_max_scaler.joblib')
            df_procesado['Examen_admision_scaled'] = scaler.transform(df_procesado[['Examen_admisión']])
            df_procesado = df_procesado.drop(columns=['Examen_admisión'], errors='ignore')
            
            # Reordenar las columnas conforme lo espera el modelo
            columnas_ordenadas = si_columnas_one_hot + ['Examen_admision_scaled']
            df_procesado = df_procesado[columnas_ordenadas]
            
            st.subheader("Datos Procesados para el Modelo")
            st.dataframe(df_procesado)
            
            # 4. Predicción con 'bagging_optimizado.joblib'
            model = joblib.load('bagging_optimizado.joblib')
            prediccion = model.predict(df_procesado)
            
            st.success(f"La predicción del modelo (Nota Final Estimada) es: {prediccion[0]:.4f}")
            
        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento o la predicción: {e}")

with tab2:
    st.header("Cargar archivo de datos")
    st.write("Sube un archivo Excel (.xlsx) que contenga al menos las columnas `Felder` y `Examen_admisión` para realizar predicciones masivas.")
    
    uploaded_file = st.file_uploader("Selecciona un archivo Excel", type=["xlsx"])
    
    if uploaded_file is not None:
        try:
            # Leer el archivo subido
            df_uploaded = pd.read_excel(uploaded_file)
            st.write("Datos cargados exitosamente. Vista previa:")
            st.dataframe(df_uploaded.head())
            
            if 'Felder' not in df_uploaded.columns or 'Examen_admisión' not in df_uploaded.columns:
                st.error("El archivo debe contener las columnas 'Felder' y 'Examen_admisión'.")
            else:
                if st.button("Procesar y Predecir Archivo", key="btn_archivo"):
                    # Copia para procesamiento
                    df_proc_masivo = df_uploaded.copy()
                    
                    # Aplicar One-Hot Encoding según el esquema guardado
                    one_hot_transformer = joblib.load('one_hot_columns.joblib')
                    
                    # Crear las columnas dummies
                    for col in one_hot_transformer:
                        if 'Felder_' in col:
                            valor_cat = col.replace('Felder_', '')
                            df_proc_masivo[col] = (df_proc_masivo['Felder'] == valor_cat).astype(int)
                    
                    # Cargar el scaler y aplicar
                    scaler = joblib.load('min_max_scaler.joblib')
                    df_proc_masivo['Examen_admision_scaled'] = scaler.transform(df_proc_masivo[['Examen_admisión']])
                    
                    # Alinear columnas con el esquema original de entrenamiento
                    columnas_modelo = [col for col in one_hot_transformer]
                    for col in columnas_modelo:
                        if col not in df_proc_masivo.columns:
                            df_proc_masivo[col] = 0
                            
                    X_final_masivo = df_proc_masivo[columnas_modelo]
                    
                    # Realizar la predicción masiva
                    model = joblib.load('bagging_optimizado.joblib')
                    predicciones_masivas = model.predict(X_final_masivo)
                    
                    # Mostrar resultados junto con el DataFrame original
                    df_uploaded['Prediccion_Nota_Final'] = predicciones_masivas
                    st.subheader("Resultados de las Predicciones")
                    st.dataframe(df_uploaded)
                    
                    # Opción para descargar los resultados
                    output_excel = df_uploaded.to_excel(index=False)
                    
                    # Generar un link/botón de descarga para el archivo Excel modificado
                    import io
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df_uploaded.to_excel(writer, sheet_name='Predicciones', index=False)
                    
                    st.download_button(
                        label="Descargar Resultados en Excel",
                        data=buffer.getvalue(),
                        file_name="predicciones_resultados.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
        except Exception as e:
            st.error(f"Ocurrió un error al procesar el archivo: {e}")
