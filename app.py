import streamlit as st
import pickle
import pandas as pd

# ---------------------------
# Load model
# ---------------------------
with open("models/lasso_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Student Grade Predictor", layout="centered")

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.title("About")
st.sidebar.write(
    """
    This app predicts a **student’s final grade (G3)**  
    using a trained **Lasso Regression model**.  

    Fill in the details and click **Predict Grade** to get the result.
    """
)

# ---------------------------
# Main Title
# ---------------------------
st.title("Student Final Grade Predictor")
st.markdown("Enter the student details below:")

# ---------------------------
# Feature Inputs (one column)
# ---------------------------
selected_features = ['G2', 'G1', 'Medu', 'Fjob_teacher', 'Mjob_other']

inputs = {}
# Numerical inputs
inputs["G2"] = st.number_input("Previous Grade G2 (0-20)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
inputs["G1"] = st.number_input("Previous Grade G1 (0-20)", min_value=0.0, max_value=20.0, value=10.0, step=0.1)
inputs["Medu"] = st.number_input("Mother's Education (0-4)", min_value=0, max_value=4, value=2, step=1)

# Boolean inputs (radio buttons instead of checkboxes)
inputs["Fjob_teacher"] = st.radio(
    "Is the father's job 'Teacher'?",
    options=["No", "Yes"],
    index=0
)
inputs["Mjob_other"] = st.radio(
    "Is the mother’s job in a sector other than health or service?",
    options=["No", "Yes"],
    index=0
)

# Convert Yes/No → True/False
inputs["Fjob_teacher"] = True if inputs["Fjob_teacher"] == "Yes" else False
inputs["Mjob_other"] = True if inputs["Mjob_other"] == "Yes" else False

input_df = pd.DataFrame([inputs])

# ---------------------------
# Prediction
# ---------------------------
st.markdown("---")
if st.button("Predict Grade"):
    prediction = model.predict(input_df)[0]

    st.markdown(
        f"""
        <div style="padding:20px; border-radius:10px; background-color:#f9f9f9; 
                    text-align:center; border:1px solid #ddd; max-width:400px; margin:auto;">
            <h3 style="color:#333;">Predicted Final Grade (G3)</h3>
            <h1 style="color:#007ACC;">{prediction:.2f}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )