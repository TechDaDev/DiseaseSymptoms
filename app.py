import warnings
# Suppress scikit-learn version warnings
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import time
import requests
from decouple import config

# --- CONFIG ---
DEEPSEEK_API_KEY = config("DEEPSEEK_API_KEY", default="sk-3cf4b6378df44ee3bd3342b8e27ec2ee")

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="BioMed AI | Disease Diagnostic",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DEEPSEEK HELPERS (Enhanced with P.R.O.M.P.T. Framework & Multilingual Support) ---
def build_explanation_prompt(
    disease,
    outcome,
    disease_conf,
    outcome_conf,
    patient_data
):
    """
    P: Purpose - Explain AI clinical prediction in English & Arabic for decision support.
    R: Role - Senior Clinical Narrative Specialist & Medical AI Interpreter (Multilingual).
    O: Output - Two distinct sections: [ENGLISH] and [ARABIC].
    M: Markers - No diagnosis, base ONLY on provided data, cautious tone.
    P: Patterns - Narrative structure focusing on Age -> Vitals -> Symptoms.
    T: Terminology - Professional, clinical, precise, non-certain.
    """
    return f"""
ROLE: You are acting as a Senior Clinical Narrative Specialist and Medical AI Interpreter fluent in both English and Arabic.

PURPOSE: Your task is to provide a technical narrative explaining why a machine learning model likely reached the following conclusion. This is for clinical decision support, NOT for direct patient diagnosis.

--- INPUT DATA ---
PREDICTED CONDITION: {disease} (Confidence: {disease_conf:.2%})
CLINICAL STATUS: {outcome} (Confidence: {outcome_conf:.2%})

PATIENT BIOMETRICS:
- Age: {patient_data['Age']}
- Gender: {patient_data['Gender']}
- Vitals: BP {patient_data['Blood Pressure']}, Cholesterol {patient_data['Cholesterol Level']}
- Active Symptoms: Fever={patient_data['Fever']}, Cough={patient_data['Cough']}, Fatigue={patient_data['Fatigue']}, Dyspnea={patient_data['Difficulty Breathing']}

CONSTRAINTS (MARKERS):
1. Use cautious, probabilistic language (e.g., "suggests," "aligns with," "may indicate").
2. DO NOT diagnose the patient.
3. DO NOT suggest treatments or medications.
4. DO NOT mention diseases not listed in the 'PREDICTED CONDITION'.
5. Base the logic ONLY on the provided biometrics and symptoms.

OUTPUT FORMAT:
Provide the explanation in TWO distinct sections.
- First section tagged with [ENGLISH]: A 5-7 sentence clinical explanation in professional English.
- Second section tagged with [ARABIC]: A professional medical translation in Arabic (Modern Standard Arabic).

Focus on how the intersection of the patient's age, specific vitals, and symptomatology weighted the model's prediction. Do not use introductory filler.
"""

def get_deepseek_explanation(prompt):
    url = "https://api.deepseek.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You explain AI predictions for clinical decision support."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Could not generate clinical explanation at this time. (Error: {str(e)})"

# --- LOAD ASSETS (New Dual Model Pipeline) ---
@st.cache_resource
def load_bio_assets():
    disease_model = joblib.load(os.path.join('Models', 'rf_disease_only.pkl'))
    outcome_model = joblib.load(os.path.join('Models', 'rf_outcome_only.pkl'))
    le_disease = joblib.load(os.path.join('Models', 'label_encoder_disease.pkl'))
    scaler = joblib.load(os.path.join('Models', 'scaler_global.pkl'))
    return disease_model, outcome_model, le_disease, scaler

try:
    rf_disease, rf_outcome, le_disease, scaler = load_bio_assets()
except Exception as e:
    st.error("Diagnostic engine components missing. Please re-run the training pipeline.")
    st.stop()

# --- PREMIUM DARK THEME CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
        font-family: 'Outfit', sans-serif;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    
    /* Header Container */
    .header-box {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 3rem;
        border-radius: 28px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        border: 1px solid #334155;
        margin-bottom: 2rem;
        text-align: center;
    }
    .header-box h1 {
        color: #0ea5e9 !important; /* Sky Blue */
        margin-bottom: 5px;
        font-size: 3.5rem !important;
    }
    .header-box p {
        color: #94a3b8 !important;
        font-size: 1.2rem;
    }

    /* Form & Input Labels - FORCE BRIGHT TEXT */
    .stForm {
        background-color: #1e293b !important;
        padding: 2.5rem !important;
        border-radius: 24px !important;
        border: 1px solid #334155 !important;
    }
    
    label, p, h3, h4, .stMarkdown, .stSelectbox label {
        color: #f1f5f9 !important;
        font-weight: 500 !important;
    }

    [data-testid="stWidgetLabel"] p {
        color: #f1f5f9 !important;
    }

    /* Button */
    .stButton > button {
        background: linear-gradient(90deg, #0ea5e9 0%, #3b82f6 100%) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 1rem 2rem !important;
        border-radius: 14px !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        width: 100% !important;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Result Panels */
    .result-card {
        padding: 35px;
        border-radius: 24px;
        text-align: center;
        margin-top: 1.5rem;
        border: 2px solid #334155;
        background: rgba(30, 41, 59, 0.5);
    }
    .disease-display {
        font-size: 3rem;
        font-weight: 800;
        color: #0ea5e9;
        margin: 10px 0;
        text-shadow: 0 0 15px rgba(14, 165, 233, 0.3);
    }
    .status-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 50px;
        font-weight: 700;
        font-size: 1.1rem;
        margin-top: 10px;
    }
    .pos-badge { background: #fee2e2; color: #991b1b; }
    .neg-badge { background: #dcfce7; color: #166534; }

    /* Explanation Styling */
    .explanation-box {
        background: #1e293b;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #334155;
        margin-top: 1rem;
        line-height: 1.6;
        color: #cbd5e1;
    }

    [data-testid="stMetricValue"] { color: #f1f5f9 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }

</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    # College Logo
    try:
        st.image("logo/coai_logo.png", width=180)
    except:
        st.image("https://cdn-icons-png.flaticon.com/512/2864/2864333.png", width=70) # Fallback
    
    st.markdown("## BioMed AI Core")
    st.markdown("---")
    
    # Student Credit
    st.info("""
    🎓 **Project Credit / تقدير المشروع**  
    
    This work was completed by first-year students in the Biomedical Applications Department at the College of Artificial Intelligence, University of Baghdad.
    """)
    
    st.markdown("""
    <div style='direction: rtl; text-align: right; background-color: rgba(14, 165, 233, 0.1); padding: 15px; border-radius: 10px; border: 1px solid rgba(14, 165, 233, 0.2); font-size: 0.9rem; color: #f1f5f9;'>
    هذا العمل تم انجازه من قبل طلاب المرحلة الأولي لقسم التطبيقات الطبية الحيوية في كلية الذكاء الاصطناعي/ جامعة بغداد
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.write("Diagnostic Hub v5.0")
    st.info("🎯 **Target:** Disease Mapping\n\n🔍 **Insight:** Dual-layer & Explainable AI.")
    st.divider()
    st.caption("Secure Healthcare Intelligence")

# --- MAIN UI ---
st.markdown("""
<div class="header-box">
    <h1>🔍 Disease Diagnostic Hub</h1>
    <p>Predicting condition status with Explainable AI Narratives</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    with st.form("diagnostic_intake"):
        st.markdown("### 📋 Patient Symptom Profile")
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            age = st.number_input("Patient Age", 0, 120, 40)
            gender = st.selectbox("Biological Gender", ["Female", "Male"])
        with row1_col2:
            bp = st.selectbox("Blood Pressure Level", ["Low", "Normal", "High"], index=1)
            chol = st.selectbox("Cholesterol Level", ["Low", "Normal", "High"], index=1)
            
        st.markdown("#### 🤒 Manifested Symptoms")
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            fever = st.checkbox("Fever")
        with s2:
            cough = st.checkbox("Cough")
        with s3:
            fatigue = st.checkbox("Fatigue")
        with s4:
            dyspnea = st.checkbox("Dyspnea")
            
        submit = st.form_submit_button("Launch Analysis Engine")

with col2:
    st.markdown("### 🧬 Analysis Protocol")
    st.write("This engine uses dual Random Forests and DeepSeek-LLM for transparent prediction narratives.")
    st.warning("📊 **Note:** Predicting 116 diseases is complex. DeepSeek will provide context for the model's choices.")
    st.divider()
    st.image("https://cdn-icons-png.flaticon.com/512/3854/3854894.png", width=300)

if submit:
    with st.spinner("Decoding diagnostic vectors..."):
        time.sleep(0.8)
        
        # Preprocessing
        binary_map = {"Male": 1, "Female": 0}
        ordinal_map = {"Low": 0, "Normal": 1, "High": 2}
        
        feature_data = pd.DataFrame([{
            "Fever": 1 if fever else 0,
            "Cough": 1 if cough else 0,
            "Fatigue": 1 if fatigue else 0,
            "Difficulty Breathing": 1 if dyspnea else 0,
            "Age": age,
            "Gender": binary_map[gender],
            "Blood Pressure": ordinal_map[bp],
            "Cholesterol Level": ordinal_map[chol]
        }])
        
        feature_data["Age"] = scaler.transform(feature_data[["Age"]])
        
        # 1. Predict Disease
        disease_idx = rf_disease.predict(feature_data)[0]
        identified_disease = le_disease.inverse_transform([disease_idx])[0]
        disease_probs = rf_disease.predict_proba(feature_data)[0]
        d_confidence = np.max(disease_probs)
        
        # 2. Predict Outcome
        outcome_idx = rf_outcome.predict(feature_data)[0]
        outcome_label = "POSITIVE" if outcome_idx == 1 else "NEGATIVE"
        outcome_probs = rf_outcome.predict_proba(feature_data)[0]
        o_confidence = outcome_probs[outcome_idx]
        
        # --- DISPLAY RESULTS ---
        st.markdown("<div class='result-card'>", unsafe_allow_html=True)
        st.write("Most Likely Underlying Condition:")
        st.markdown(f"<div class='disease-display'>{identified_disease}</div>", unsafe_allow_html=True)
        
        badge_class = "pos-badge" if outcome_label == "POSITIVE" else "neg-badge"
        st.markdown(f"Clinical Status: <span class='status-badge {badge_class}'>{outcome_label}</span>", unsafe_allow_html=True)
        
        st.divider()
        m1, m2 = st.columns(2)
        m1.metric("Disease Prob. Match", f"{d_confidence:.2%}")
        m2.metric("Status Confidence", f"{o_confidence:.2%}")
        st.progress(o_confidence, text="Diagnostic Signal Strength")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- DEEPSEEK EXPLANATION ---
        patient_summary = {
            "Age": age,
            "Gender": gender,
            "Fever": "Yes" if fever else "No",
            "Cough": "Yes" if cough else "No",
            "Fatigue": "Yes" if fatigue else "No",
            "Difficulty Breathing": "Yes" if dyspnea else "No",
            "Blood Pressure": bp,
            "Cholesterol Level": chol
        }

        prompt = build_explanation_prompt(
            identified_disease,
            outcome_label,
            d_confidence,
            o_confidence,
            patient_summary
        )

        with st.spinner("DeepSeek generating clinical explanation..."):
            explanation = get_deepseek_explanation(prompt)

        # Parse Multilingual Response
        eng_part = ""
        ara_part = ""
        if "[ENGLISH]" in explanation and "[ARABIC]" in explanation:
            parts = explanation.split("[ARABIC]")
            eng_part = parts[0].replace("[ENGLISH]", "").strip()
            ara_part = parts[1].strip()
        else:
            eng_part = explanation # Fallback

        st.markdown("### 🧠 AI Clinical Narrative")
        tab_eng, tab_ara = st.tabs(["🇺🇸 English Explanation", "🇸🇦 Arabic Explanation"])
        
        with tab_eng:
            st.markdown(f"<div class='explanation-box'>{eng_part}</div>", unsafe_allow_html=True)
        
        with tab_ara:
            st.markdown(f"<div class='explanation-box' style='direction: rtl; text-align: right;'>{ara_part}</div>", unsafe_allow_html=True)
        
        st.caption(
            "⚠️ This explanation is generated by a language model to interpret the AI prediction. "
            "It does not constitute medical advice or diagnosis. / هذه التوضيحات تم إنشاؤها بواسطة نموذج لغوي لتفسير توقعات الذكاء الاصطناعي ولا تعتبر استشارة طبية."
        )

# --- FOOTER ---
st.markdown("<br><br><p style='text-align: center; color: #64748b; font-size: 0.8rem;'>BioMed Diagnostic Hub v5.0 | Powered by Random Forest & DeepSeek LLM</p>", unsafe_allow_html=True)
