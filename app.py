import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺")


@st.cache_resource
def load_files():
    return joblib.load("model(1).pkl"), joblib.load("scaler(1).pkl")


model, scaler = load_files()

COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

DEFAULTS = {"Pregnancies": 1, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20,
            "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30}
HIGH_RISK = {"Pregnancies": 6, "Glucose": 180, "BloodPressure": 85, "SkinThickness": 35,
             "Insulin": 200, "BMI": 38.0, "DiabetesPedigreeFunction": 1.0, "Age": 55}
LOW_RISK = {"Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 20,
            "Insulin": 70, "BMI": 22.0, "DiabetesPedigreeFunction": 0.2, "Age": 25}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)


def fill(values):
    for k, v in values.items():
        st.session_state[k] = v


def predict(df):
    return model.predict_proba(scaler.transform(df[COLS]))[:, 1]


st.title("🩺 Diabetes Risk Predictor")
st.warning("Learning project only. This is NOT a medical diagnosis. "
           "Please consult a doctor for real medical advice.")

tab1, tab2 = st.tabs(["Single patient", "Batch (CSV upload)"])

# ---------- Tab 1: single prediction ----------
with tab1:
    b1, b2 = st.columns(2)
    b1.button("Load high-risk example", on_click=fill, args=(HIGH_RISK,))
    b2.button("Load low-risk example", on_click=fill, args=(LOW_RISK,))

    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Pregnancies", 0, 20, key="Pregnancies")
        st.number_input("Glucose (mg/dL)", 40, 250, key="Glucose")
        st.number_input("Blood Pressure (mm Hg)", 30, 130, key="BloodPressure")
        st.number_input("Skin Thickness (mm)", 5, 100, key="SkinThickness")
    with c2:
        st.number_input("Insulin (mu U/ml)", 10, 900, key="Insulin")
        st.number_input("BMI", 10.0, 70.0, key="BMI")
        st.number_input("Diabetes Pedigree Function", 0.05, 2.5, key="DiabetesPedigreeFunction")
        st.number_input("Age", 18, 100, key="Age")

    if st.button("Predict", type="primary"):
        row = pd.DataFrame([[st.session_state[c] for c in COLS]], columns=COLS)
        prob = predict(row)[0]
        st.progress(float(prob))

        if prob >= 0.5:
            st.error(f"Higher risk of diabetes: {prob:.0%}")
            st.write("Please consult a doctor for a proper blood test "
                     "(fasting glucose or HbA1c). General tips: balanced diet, "
                     "regular exercise, healthy weight, regular checkups.")
        else:
            st.success(f"Lower risk of diabetes: {prob:.0%}")
            st.write("Keep up a healthy lifestyle and get regular checkups.")

        st.subheader("Which features matter most to the model?")
        importance = pd.Series(model.feature_importances_, index=COLS).sort_values()
        st.bar_chart(importance)

# ---------- Tab 2: batch prediction ----------
with tab2:
    st.write("Upload a CSV with these columns (any order):")
    st.code(", ".join(COLS))

    sample = pd.DataFrame([HIGH_RISK, LOW_RISK])[COLS]
    st.download_button("Download sample CSV", sample.to_csv(index=False),
                       "sample_patients.csv", "text/csv")

    file = st.file_uploader("Upload CSV", type="csv")
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["Risk %"] = (predict(data) * 100).round(1)
            data["Prediction"] = data["Risk %"].apply(
                lambda r: "Higher risk" if r >= 50 else "Lower risk")
            st.dataframe(data)
            st.download_button("Download results", data.to_csv(index=False),
                               "predictions.csv", "text/csv")
