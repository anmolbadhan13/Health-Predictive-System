import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the model and the scaler
def load_model_and_scaler():
    model = joblib.load('front.pkl')  # Replace with your actual model file name
    scaler = joblib.load('scaler.pkl')         # Replace with your actual scaler file name
    return model, scaler

model, scaler = load_model_and_scaler()

# Function to predict diabetes probability
def predict_diabetes(user_input_df):
    input_scaled = scaler.transform(user_input_df)
    prediction_proba = model.predict_proba(input_scaled)
    return prediction_proba[0][1]

# Function to provide feedback based on input values
def provide_feedback(bmi, glucose, blood_pressure):
    feedback = {
        'BMI': "Underweight" if bmi < 18.5 else "Normal weight" if bmi <= 24.9 else "Overweight" if bmi <= 29.9 else "Obese",
        'Glucose': "Low" if glucose < 70 else "Normal" if glucose <= 99 else "High (Prediabetic)" if glucose <= 125 else "Very High (Diabetic)",
        'Blood Pressure': "Normal" if blood_pressure < 120 else "Elevated" if blood_pressure <= 139 else "High"
    }
    return feedback

# Function to plot health metrics
def plot_health_metrics(bmi, glucose, blood_pressure):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    categories = [
        ['Underweight', 'Normal', 'Overweight', 'Obese'],
        ['Low', 'Normal', 'High (Prediabetic)', 'Very High (Diabetic)'],
        ['Normal', 'Elevated', 'High']
    ]
    thresholds = [
        [18.5, 24.9, 29.9, 50],  # BMI thresholds
        [70, 99, 125, 200],  # Glucose thresholds
        [120, 139, 150]  # Blood Pressure thresholds
    ]
    user_values = [bmi, glucose, blood_pressure]
    colors = ['skyblue', 'lightgreen', 'salmon']
    titles = ['BMI', 'Glucose Levels', 'Blood Pressure Levels']

    for i in range(3):
        ax[i].bar(categories[i], thresholds[i], color=colors[i])

        # Draw the user's value as a point
        ax[i].scatter([user_values[i]], [thresholds[i][-1]], color='purple', zorder=5, label=f'Your {titles[i]}: {user_values[i]}')

        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Categories')
        ax[i].set_ylabel('Values')
        ax[i].legend()

    st.pyplot(fig)

# Main Streamlit application function
def diabetes_app():
    st.title('🩺 Diabetes Prediction Tool')

    # Create a two-column layout with a smaller column for the sidebar
    sidebar_col, divider_col, content_col = st.columns([1.5, 0.1, 4], gap="medium")

    with sidebar_col:
        # Sidebar with input sliders
        st.subheader("User Input")
        pregnancies = st.slider('Number of Pregnancies', 0, 17, 0)
        glucose = st.slider('Glucose Level', 50, 200, 100)
        blood_pressure = st.slider('Blood Pressure', 40, 130, 70)
        skin_thickness = st.slider('Skin Thickness', 10, 100, 20)
        insulin = st.slider('Insulin Level', 15, 276, 30)
        bmi = st.slider('BMI (Body Mass Index)', 15.0, 50.0, 25.0)
        dpf = st.slider('Diabetes Pedigree Function', 0.0, 2.5, 0.5)
        age = st.slider('Age', 18, 100, 30)

    with content_col:
        # Prepare user input for the model
        user_input_df = pd.DataFrame({
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        })

        # Predict diabetes probability
        diabetes_probability = predict_diabetes(user_input_df)
        type_2_diabetes_probability = diabetes_probability * 0.7  # Example adjustment
        feedback = provide_feedback(bmi, glucose, blood_pressure)

        # Displaying the prediction results and health metrics in separate containers
        with st.container():
            st.subheader('👤 Your Health Metrics:')
            st.json({
                "Age": age,
                "BMI": f"{bmi} - {feedback['BMI']}",
                "Glucose Level": f"{glucose} mg/dL - {feedback['Glucose']}",
                "Blood Pressure": f"{blood_pressure} mmHg - {feedback['Blood Pressure']}"
            })

        with st.container():
            st.subheader('🔍 Diabetes Assessment')
            # Display each metric in its own row
            st.metric("Probability of Diabetes", f"{diabetes_probability * 100:.2f}%")
            st.metric("Probability of Type 2 Diabetes", f"{type_2_diabetes_probability * 100:.2f}%")
            status_color = "🔴" if diabetes_probability > 0.5 else "🟢"
            st.metric("Diagnosis", f"{status_color} {'Diabetic' if diabetes_probability > 0.5 else 'Not Diabetic'}")

    st.subheader('📈 Visual Health Analysis')
    plot_health_metrics(bmi, glucose, blood_pressure)

# Run the app
if _name_ == '_main_':
  diabetes_app()