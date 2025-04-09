import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Mobile Price Predictor Pro", layout="wide")

# Chargement du modèle ET du scaler
@st.cache_resource
def load_artifacts():
    try:
        # Chargement avec joblib
        artifacts = joblib.load('mobile_price_classifier.joblib')
            
        # Vérification du contenu
        if isinstance(artifacts, dict):
            model = artifacts.get('model')
            scaler = artifacts.get('scaler')
        else:
            model = artifacts
            scaler = None
        
        if not hasattr(model, 'predict'):
            raise ValueError("Le fichier ne contient pas un modèle valide")
            
        return model, scaler
    except Exception as e:
        st.error(f"ERREUR DE CHARGEMENT : {str(e)}")
        return None, None

model, scaler = load_artifacts()

# Liste des caractéristiques (garder le même ordre que pendant l'entraînement)
features = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 
    'pc', 'px_height', 'px_width', 'ram', 'sc_h', 
    'sc_w', 'talk_time', 'three_g', 'touch_screen', 'wifi'
]

# Dictionnaire des gammes de prix
price_ranges = {
    0: ("Low Cost", "green", "💰"),
    1: ("Medium Cost", "blue", "💰💰"),
    2: ("High Cost", "orange", "💰💰💰"),
    3: ("Very High Cost", "red", "💰💰💰💰")
}

# Interface utilisateur
st.title("📱 Mobile Price Predictor Pro")

def get_user_input():
    inputs = {}
    with st.form("input_form"):
        cols = st.columns(3)
        
        with cols[0]:
            inputs['battery_power'] = st.slider('Battery (mAh)', 500, 2000, 1000)
            inputs['blue'] = st.selectbox("Bluetooth", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            inputs['clock_speed'] = st.slider('CPU Speed (GHz)', 0.5, 3.0, 1.0)
            inputs['dual_sim'] = st.selectbox("Dual SIM", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            inputs['fc'] = st.slider('Front Camera', 0, 20, 5)
        
        with cols[1]:
            inputs['four_g'] = st.selectbox("4G", [0, 1], format_func=lambda x: "Oui" if x else "Non")
            inputs['int_memory'] = st.slider('Storage (GB)', 2, 64, 16)
            inputs['m_dep'] = st.slider('Thickness (cm)', 0.1, 1.0, 0.5)
            inputs['mobile_wt'] = st.slider('Weight (g)', 80, 200, 150)
            inputs['n_cores'] = st.slider('CPU Cores', 1, 8, 4)
        
        with cols[2]:
            inputs['pc'] = st.slider('Main Camera', 0, 20, 8)
            inputs['px_height'] = st.number_input('Pixel Height', 0, 2000, 1000)
            inputs['px_width'] = st.number_input('Pixel Width', 0, 2000, 1000)
            inputs['ram'] = st.slider('RAM (MB)', 256, 4096, 2048)
            inputs['sc_h'] = st.slider('Screen Height (cm)', 5, 20, 10)
        
        inputs['sc_w'] = st.slider('Screen Width (cm)', 5, 20, 10)
        inputs['talk_time'] = st.slider('Talk Time (h)', 2, 20, 10)
        inputs['three_g'] = st.selectbox("3G", [0, 1])
        inputs['touch_screen'] = st.selectbox("Touchscreen", [0, 1])
        inputs['wifi'] = st.selectbox("Wi-Fi", [0, 1])
        
        submitted = st.form_submit_button("Prédire le prix")
    
    if submitted:
        return pd.DataFrame([inputs], columns=features)

# Main execution
user_input = get_user_input()

if user_input is not None and model is not None:
    with st.spinner('Analyse en cours...'):
        try:
            # Appliquer la même transformation qu'à l'entraînement
            if scaler is not None:
                user_input_scaled = scaler.transform(user_input)
                user_input = pd.DataFrame(user_input_scaled, columns=features)
            
            # Debug
            st.write("Données envoyées au modèle:", user_input)
            
            # Prédiction
            prediction = model.predict(user_input)[0]
            label, color, emoji = price_ranges[prediction]
            
            # Affichage
            st.success(f"Prédiction : {emoji} {label}")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(user_input)[0]
                st.write("Probabilités:", {k: f"{v:.2%}" for k, v in enumerate(proba)})
        
        except Exception as e:
            st.error(f"Erreur : {str(e)}")

# Section de diagnostic
with st.expander("Diagnostic Avancé"):
    # Test avec valeurs minimales (devrait donner Low Cost)
    test_data = {
        'battery_power': 500, 'blue': 0, 'clock_speed': 0.5, 'dual_sim': 0,
        'fc': 0, 'four_g': 0, 'int_memory': 2, 'm_dep': 0.1, 'mobile_wt': 80,
        'n_cores': 1, 'pc': 0, 'px_height': 500, 'px_width': 500, 'ram': 256,
        'sc_h': 5, 'sc_w': 5, 'talk_time': 2, 'three_g': 0, 'touch_screen': 0, 'wifi': 0
    }
    
    if st.button("Tester avec valeurs Low Cost"):
        test_df = pd.DataFrame([test_data], columns=features)
        if scaler is not None:
            test_df = pd.DataFrame(scaler.transform(test_df), columns=features)
        
        try:
            pred = model.predict(test_df)[0]
            st.write(f"Résultat: {price_ranges[pred][0]} (devrait être Low Cost)")
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(test_df)[0]
                st.write("Probabilités:", {k: f"{v:.2%}" for k, v in enumerate(proba)})
        except Exception as e:
            st.error(f"Erreur lors du test: {str(e)}")