import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("survey.csv")

df = load_data()

# Load model and preprocessing tools
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
model_columns = joblib.load("model_columns.pkl")

# Function to clean gender
def clean_gender(g):
    g = g.lower()
    if 'male' in g:
        return 'Male'
    elif 'female' in g:
        return 'Female'
    else:
        return 'Other'

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "EDA", "Prediction", "Conclusion"])

if page == "Introduction":
    st.title("Mental Health Treatment Prediction")
    st.markdown("""
    This app predicts whether a person is likely to seek mental health treatment based on survey responses.
    It includes EDA, preprocessing, and a machine learning model for real-time predictions.
    """)

elif page == "EDA":
    st.title("Exploratory Data Analysis")

    if st.checkbox("Show raw data"):
        st.dataframe(df)

    if st.checkbox("Summary Statistics"):
        st.write(df.describe(include='all'))

    if st.checkbox("Missing Values Heatmap"):
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False)
        st.pyplot(fig)

    if st.checkbox("Age Distribution"):
        fig = plt.figure()
        sns.histplot(df['Age'], bins=30)
        st.pyplot(fig)

    if st.checkbox("Correlation Heatmap"):
        num_df = df.select_dtypes(include=['int64', 'float64'])
        fig = plt.figure(figsize=(10, 6))
        sns.heatmap(num_df.corr(), annot=True)
        st.pyplot(fig)

elif page == "Prediction":
    st.title("Make a Prediction")

    age = st.slider("Age", 18, 80, 30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    remote_work = st.selectbox("Remote Work", ['Yes', 'No'])
    family_history = st.selectbox("Family History of Mental Illness", ['Yes', 'No'])
    work_interfere = st.selectbox("Work Interference", ['Never', 'Rarely', 'Sometimes', 'Often'])
    benefits = st.selectbox("Mental Health Benefits", ['Yes', 'No', "Don't know"])
    anonymity = st.selectbox("Anonymity Protected?", ['Yes', 'No', "Don't know"])
    leave = st.selectbox("Ease of Taking Leave", ['Very easy', 'Somewhat easy', 'Somewhat difficult', 'Very difficult'])

    input_data = {
        'Age': [age],
        'Gender': [clean_gender(gender)],
        'remote_work': [remote_work],
        'family_history': [family_history],
        'work_interfere': [work_interfere],
        'benefits': [benefits],
        'anonymity': [anonymity],
        'leave': [leave]
    }

    input_df = pd.DataFrame(input_data)

    for col in encoders:
        input_df[col] = encoders[col].transform(input_df[col])

    input_df['Age'] = scaler.transform(input_df[['Age']])
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)[0]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success("The person is likely to seek mental health treatment.")
    else:
        st.warning("The person is unlikely to seek mental health treatment.")

elif page == "Conclusion":
    st.title("Conclusion")
    st.markdown("""
    - This project analyzed mental health data using various techniques.
    - We trained a classification model that predicts treatment-seeking behavior.
    - The Streamlit app provides interactive EDA and real-time prediction.
    """)
