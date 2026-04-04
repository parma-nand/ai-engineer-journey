import streamlit as st
import requests

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write("Enter passenger details to predict survival.")

# Input fields
age = st.slider("Age", min_value=1, max_value=100, value=25)

sex = st.selectbox("Sex", options=["Male", "Female"])
sex_val = 0 if sex == "Male" else 1

pclass = st.selectbox("Passenger Class", options=[1, 2, 3], format_func=lambda x: f"Class {x}")

fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)

embarked = st.selectbox("Embarked", options=["S - Southampton", "Q - Queenstown", "C - Cherbourg"])
embarked_val = {"S - Southampton": 0, "Q - Queenstown": 1, "C - Cherbourg": 2}[embarked]

st.divider()

# Predict button
if st.button("🔍 Predict Survival"):
    payload = {
        "Age": age,
        "Sex": sex_val,
        "Pclass": pclass,
        "Fare": fare,
        "Embarked": embarked_val
    }

    try:
        response = requests.post("http://titanic-api:8000/predict", json=payload)
        result = response.json()

        if result["survived"] == 1:
            st.success(f"✅ **Survived!**  Probability: `{result['probability']}`")
            st.balloons()
        else:
            st.error(f"❌ **Did not survive.**  Probability: `{result['probability']}`")

    except Exception as e:
        st.error(f"Could not connect to API: {e}")