import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import ExtraTreesClassifier

# Page configuration
st.set_page_config(page_title="Customer Loyalty Predictor", layout="wide")

# Title
st.title("Customer Loyalty Prediction Tool")
st.markdown("""
This app predicts whether a customer will be loyal (1) or not loyal (0) based on their characteristics and behaviors.
""")

# Model selection
model_option = st.radio(
    "Select the model to use:",
    ("Extra_tress_classifier.pkl", "best_Extra_tress_classifier.pkl"),
    horizontal=True
)

# Load the selected model
@st.cache_resource
def load_model(model_path):
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model(model_option)

# Input fields organized in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Customer Demographics")
    Genero = st.selectbox("Gender", ["Male", "Female"])
    Mascota = st.selectbox("Has Pet", ["Yes", "No"])
    Lealtad_Cliente = st.selectbox("Current Loyalty Status", ["Loyal", "Not Loyal"])
    Cliente_frecuen = st.selectbox("Frequent Client", ["Yes", "No"])
    Regresa_clinic = st.selectbox("Returns to Clinic", ["Yes", "No"])
    
    st.subheader("Service Perception")
    Serv_necesarios = st.slider("Necessary Services", 1, 5, 3)
    Atenc_pers = st.slider("Personal Attention", 1, 5, 3)
    Confi_segur = st.slider("Trust/Security", 1, 5, 3)

with col2:
    st.subheader("Professional Evaluation")
    Prof_alt_capac = st.slider("High Capacity Professionals", 1, 5, 3)
    Medico_carism = st.slider("Charismatic Doctor", 1, 5, 3)
    Inspira_confianz = st.slider("Inspires Confidence", 1, 5, 3)
    Conoce_profund = st.slider("Deep Knowledge", 1, 5, 3)
    Buen_trato_asist = st.slider("Good Assistant Treatment", 1, 5, 3)
    Recon_logro_asist = st.slider("Recognizes Assistant Achievements", 1, 5, 3)
    Corrig_error = st.slider("Corrects Mistakes", 1, 5, 3)
    
    st.subheader("Location Factors")
    Cerca_viv = st.slider("Close to Home", 1, 5, 3)
    Infraes_atract = st.slider("Attractive Infrastructure", 1, 5, 3)

with col3:
    st.subheader("Price Perception")
    Precio_acces = st.slider("Accessible Prices", 1, 5, 3)
    Excelent_calid_precio = st.slider("Excellent Quality-Price Ratio", 1, 5, 3)
    
    st.subheader("Social Media Interaction")
    Conce_redes_clinic = st.selectbox("Knows Clinic's Social Media", ["Yes", "No"])
    Sigue_redes_clinic = st.selectbox("Follows Clinic's Social Media", ["Yes", "No"])
    Comenta_redes_clinic = st.selectbox("Comments on Clinic's Social Media", ["Yes", "No"])
    
    st.subheader("Payment Disposition")
    Disposic_pago_vacunas = st.slider("Willing to Pay for Vaccines", 1, 5, 3)
    Disposic_pago_consult = st.slider("Willing to Pay for Consultations", 1, 5, 3)
    Disposic_pago_esteriliz = st.slider("Willing to Pay for Sterilization", 1, 5, 3)
    Disposic_pago_baño = st.slider("Willing to Pay for Bathing", 1, 5, 3)

# Additional payment disposition fields (couldn't fit in 3 columns)
st.subheader("Additional Payment Disposition")
col4, col5, col6 = st.columns(3)

with col4:
    Disposic_pago_cortepelo = st.slider("Willing to Pay for Haircut", 1, 5, 3)
    Disposic_pago_corteuna = st.slider("Willing to Pay for Nail Cutting", 1, 5, 3)

with col5:
    Disposic_pago_despar_inter = st.slider("Willing to Pay for Internal Deworming", 1, 5, 3)
    Disposic_pago_despar_exter = st.slider("Willing to Pay for External Deworming", 1, 5, 3)

with col6:
    Disposic_pago_accesor = st.slider("Willing to Pay for Accessories", 1, 5, 3)
    Disposic_pago_comida = st.slider("Willing to Pay for Food", 1, 5, 3)

col7, col8 = st.columns(2)

with col7:
    Disposic_pago_ecograf = st.slider("Willing to Pay for Ultrasound", 1, 5, 3)
    Disposic_pago_radiograf = st.slider("Willing to Pay for X-rays", 1, 5, 3)

with col8:
    Disposic_pago_analab = st.slider("Willing to Pay for Lab Analysis", 1, 5, 3)

# Prepare the input data
def prepare_input():
    input_data = {
        'Genero': 1 if Genero == "Male" else 0,
        'Mascota': 1 if Mascota == "Yes" else 0,
        'Lealtad_Cliente': 1 if Lealtad_Cliente == "Loyal" else 0,
        'Serv_necesarios': Serv_necesarios,
        'Atenc_pers': Atenc_pers,
        'Confi_segur': Confi_segur,
        'Prof_alt_capac': Prof_alt_capac,
        'Cerca_viv': Cerca_viv,
        'Infraes_atract': Infraes_atract,
        'Precio_acces': Precio_acces,
        'Excelent_calid_precio': Excelent_calid_precio,
        'Medico_carism': Medico_carism,
        'Inspira_confianz': Inspira_confianz,
        'Conoce_profund': Conoce_profund,
        'Buen_trato_asist': Buen_trato_asist,
        'Recon_logro_asist': Recon_logro_asist,
        'Corrig_error': Corrig_error,
        'Disposic_pago_vacunas': Disposic_pago_vacunas,
        'Disposic_pago_consult': Disposic_pago_consult,
        'Disposic_pago_esteriliz': Disposic_pago_esteriliz,
        'Disposic_pago_baño': Disposic_pago_baño,
        'Disposic_pago_cortepelo': Disposic_pago_cortepelo,
        'Disposic_pago_corteuna': Disposic_pago_corteuna,
        'Disposic_pago_despar_inter': Disposic_pago_despar_inter,
        'Disposic_pago_despar_exter': Disposic_pago_despar_exter,
        'Disposic_pago_accesor': Disposic_pago_accesor,
        'Disposic_pago_comida': Disposic_pago_comida,
        'Disposic_pago_ecograf': Disposic_pago_ecograf,
        'Disposic_pago_radiograf': Disposic_pago_radiograf,
        'Disposic_pago_analab': Disposic_pago_analab,
        'Cliente_frecuen': 1 if Cliente_frecuen == "Yes" else 0,
        'Regresa_clinic': 1 if Regresa_clinic == "Yes" else 0,
        'Conce_redes_clinic': 1 if Conce_redes_clinic == "Yes" else 0,
        'Sigue_redes_clinic': 1 if Sigue_redes_clinic == "Yes" else 0,
        'Comenta_redes_clinic': 1 if Comenta_redes_clinic == "Yes" else 0
    }
    return pd.DataFrame([input_data])

# Prediction button
if st.button("Predict Loyalty"):
    if model is not None:
        try:
            input_df = prepare_input()
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success("Prediction: Loyal (1)")
            else:
                st.error("Prediction: Not Loyal (0)")
            
            st.write(f"Probability of being Loyal: {probability[0][1]:.2f}")
            st.write(f"Probability of being Not Loyal: {probability[0][0]:.2f}")
            
            # Show input data
            st.subheader("Input Data Summary")
            st.dataframe(input_df.T.rename(columns={0: "Value"}))
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model not loaded properly. Please check the model files.")
