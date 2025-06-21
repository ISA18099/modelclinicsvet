import streamlit as st
import pandas as pd
import joblib  # Más eficiente que pickle para modelos grandes
import time
from sklearn.ensemble import ExtraTreesClassifier

# =================== CONFIGURACIÓN INICIAL ===================
st.set_page_config(
    page_title="Predicción de Lealtad - Optimizado",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =================== CACHÉ DE MODELOS ===================
@st.cache_resource(max_entries=2)  # Cachea solo los 2 modelos
def load_model(model_name):
    try:
        model_path = f"{model_name}.joblib"  # Usar joblib comprimido
        with st.spinner(f"Cargando {model_name}..."):
            model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error cargando {model_name}: {str(e)}")
        return None

# =================== INTERFAZ DE USUARIO ===================
def show_input_form():
    with st.form("input_form"):
        cols = st.columns(3)
        
        # Columna 1: Datos demográficos
        with cols[0]:
            st.subheader("📊 Demográficos")
            genero = st.selectbox("Género", ["Femenino", "Masculino"], key="genero")
            mascota = st.selectbox("Tiene Mascota", ["Sí", "No"], key="mascota")
            cliente_frecuente = st.radio("Cliente Frecuente", ["Sí", "No"], key="frecuente")

        # Columna 2: Servicios (usando sliders optimizados)
        with cols[1]:
            st.subheader("🛎️ Servicios")
            serv_necesarios = st.slider("Servicios Necesarios", 1, 5, 3, key="serv")
            atencion_personal = st.slider("Atención Personal", 1, 5, 3, key="atencion")
            precio_accesible = st.slider("Precio Accesible", 1, 5, 3, key="precio")

        # Columna 3: Redes Sociales
        with cols[2]:
            st.subheader("📱 Redes Sociales")
            conoce_redes = st.radio("Conoce Redes", ["Sí", "No"], key="conoce")
            sigue_redes = st.radio("Sigue Redes", ["Sí", "No"], key="sigue")

        # Botón de predicción con estado
        submitted = st.form_submit_button("🔍 Predecir Lealtad", type="primary")
        
    return submitted, {
        "Genero": 0 if genero == "Femenino" else 1,
        "Mascota": 1 if mascota == "Sí" else 0,
        "Cliente_frecuen": 1 if cliente_frecuente == "Sí" else 0,
        "Serv_necesarios": serv_necesarios,
        "Atenc_pers": atencion_personal,
        "Precio_acces": precio_accesible,
        "Conce_redes_clinic": 1 if conoce_redes == "Sí" else 0,
        "Sigue_redes_clinic": 1 if sigue_redes == "Sí" else 0,
        # Valores por defecto para otras features (ajustar según necesidad)
        **{k: 3 for k in [
            'Lealtad_Cliente', 'Confi_segur', 'Prof_alt_capac', 'Cerca_viv', 
            'Infraes_atract', 'Excelent_calid_precio', 'Medico_carism'
        ]}
    }

# =================== PREDICCIÓN OPTIMIZADA ===================
def make_prediction(model, input_data):
    start_time = time.time()
    try:
        df = pd.DataFrame([input_data])
        proba = model.predict_proba(df)[0]
        pred = model.predict(df)[0]
        return {
            "prediction": pred,
            "probability_loyal": proba[1],
            "time": time.time() - start_time
        }
    except Exception as e:
        st.error(f"Error en predicción: {str(e)}")
        return None

# =================== FLUJO PRINCIPAL ===================
def main():
    st.title("🚀 Predictor de Lealtad Optimizado")
    
    # Selector de modelo con caché
    model_name = st.radio(
        "Seleccione modelo:",
        ("Extra_tress_classifier", "best_Extra_tress_classifier"),
        horizontal=True
    )
    
    # Carga diferida del modelo
    if "model" not in st.session_state or st.session_state.model_name != model_name:
        with st.spinner(f"Cargando modelo {model_name}..."):
            st.session_state.model = load_model(model_name)
            st.session_state.model_name = model_name
    
    # Formulario optimizado
    submitted, input_data = show_input_form()
    
    # Predicción al enviar
    if submitted and st.session_state.model:
        result = make_prediction(st.session_state.model, input_data)
        if result:
            st.success(f"""
                **Resultado:** {"Lealtad ✅" if result['prediction'] else "No Lealtad ❌"}  
                **Probabilidad:** {result['probability_loyal']:.2%}  
                **Tiempo:** {result['time']:.3f} segundos
            """)
            
            # Debug: Mostrar solo 5 features clave
            st.json({k: input_data[k] for k in list(input_data)[:5]})

if __name__ == "__main__":
    main()
