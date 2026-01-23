# 🔍 BioMed AI | Disease Diagnostic Hub

A premium, AI-powered disease diagnostic and explanation platform developed by students at the **College of Artificial Intelligence, University of Baghdad**.

This project leverages Dual Random Forest models and the DeepSeek-LLM to provide not only predictions for over 100 diseases but also human-readable, multilingual (English & Arabic) clinical narratives explaining those predictions.

## ✨ Features

- **Dual-Model Prediction Engine**: Utilizes two distinct Random Forest models to predict both the specific disease condition and the clinical outcome (Positive/Negative).
- **Explainable AI (XAI)**: Integrated with **DeepSeek-LLM** to generate clinical explanations based on patient biometrics and symptoms.
- **Multilingual Support**: Explanations are automatically provided in both **English** and **Arabic**, making it accessible for diverse clinical settings.
- **Premium UI/UX**: A sleek, dark-themed dashboard built with Streamlit, featuring glassmorphic elements and smooth interactions.
- **Data-Driven Insights**: Trained on patient symptom profiles, including age, gender, blood pressure, cholesterol, and specific symptoms like fever, cough, and fatigue.

## 🛠️ Technology Stack

- **Machine Learning**: Scikit-learn (Random Forest)
- **Frontend**: Streamlit
- **LLM Integration**: DeepSeek API (v1)
- **Data Handling**: Pandas, Numpy
- **Environment Management**: Python-decouple (for API security)

## 📁 Project Structure

- `app.py`: The main Streamlit application.
- `Scripts/`: Data preprocessing and model training pipeline.
  - `step1_inspection.py`: Data exploration.
  - `step2_preprocessing.py`: Feature engineering and scaling.
  - `step3_modeling.py`: Model training scripts.
  - `step4_explainability.py`: LLM integration logic.
- `Models/`: Serialized Random Forest models and encoders.
- `Data/`: Project datasets.
- `logo/`: Application branding assets.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- A DeepSeek API Key

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/TechDaDev/DiseaseSymptoms.git
   cd DiseaseSymptoms
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your DeepSeek API key:
   ```env
   DEEPSEEK_API_KEY=your_actual_api_key_here
   ```

4. **Run the application**:
   ```bash
   streamlit run app.py
   ```

## 🎓 Academic Credit

This project was developed by first-year students in the **Biomedical Applications Department** at the **College of Artificial Intelligence, University of Baghdad**.

> "هذا العمل تم انجازه من قبل طلاب المرحلة الأولي لقسم التطبيقات الطبية الحيوية في كلية الذكاء الاصطناعي/ جامعة بغداد"

---
**Disclaimer**: This application is for educational and clinical decision support purposes only. It does not constitute medical advice or diagnosis.
