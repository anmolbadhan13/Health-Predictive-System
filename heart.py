import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load the model
def load_model():
    model = joblib.load('heart.pkl')  # Replace with your actual model file name
    return model

model = load_model()

# Function to predict heart disease probability
def predict_heart_disease(user_input_df):
    # Use the model to predict probability directly on the input data
    prediction_proba = model.predict_proba(user_input_df)
    return prediction_proba[0][1]

# Function to provide feedback based on input values
def provide_feedback(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    feedback = {
        'Cholesterol': "Normal" if chol < 200 else "Borderline High" if chol <= 239 else "High",
        'Max Heart Rate': "Normal" if thalach >= 120 else "Low",
        'Resting Blood Pressure': "Normal" if trestbps < 120 else "Elevated" if trestbps <= 129 else "High"
    }
    return feedback

# Function to plot health metrics
def plot_health_metrics(chol, thalach, trestbps):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # Define categories and values for each metric
    categories = [
        ['Normal (<200)', 'Borderline High (200-239)', 'High (240+)'],
        ['Normal (>=120)', 'Low (<120)'],
        ['Normal (<120)', 'Elevated (120-129)', 'High (130+)']
    ]
    thresholds = [
        [200, 239, 300],  # Cholesterol thresholds
        [120, 200],       # Max Heart Rate thresholds
        [120, 129, 200]   # Resting Blood Pressure thresholds
    ]
    user_values = [chol, thalach, trestbps]
    colors = ['skyblue', 'lightgreen', 'salmon']
    titles = ['Cholesterol', 'Max Heart Rate', 'Resting Blood Pressure']

    for i in range(3):
        # Draw bars for categories with width defined by thresholds
        ax[i].barh(categories[i], thresholds[i], color=colors[i])

        # Add a vertical line for the user's value
        ax[i].axvline(x=user_values[i], color='purple', linewidth=2, linestyle='--', label=f'Your {titles[i]}: {user_values[i]}')

        ax[i].set_title(titles[i])
        ax[i].set_xlabel('Values')
        ax[i].set_ylabel('Categories')

        # Add legend
        ax[i].legend()

    st.pyplot(fig)

# Main Streamlit application function
def heart_disease_app():
    st.title('❤ Heart Disease Prediction Tool')

    # Create a two-column layout with a smaller column for the sidebar
    sidebar_col, divider_col, content_col = st.columns([1.5, 0.1, 4], gap="medium")

    with sidebar_col:
        # Sidebar with input sliders
        st.subheader("User Input")
        age = st.slider('Age', 18, 100, 50)
        sex = st.radio('Sex', ['Male', 'Female'])
        cp = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
        trestbps = st.slider('Resting Blood Pressure (mmHg)', 80, 200, 120)
        chol = st.slider('Cholesterol Level (mg/dL)', 100, 500, 200)
        fbs = st.radio('Fasting Blood Sugar > 120 mg/dL?', ['Yes', 'No'])
        restecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Probable left ventricular hypertrophy'])
        thalach = st.slider('Max Heart Rate Achieved (bpm)', 60, 200, 150)
        exang = st.radio('Exercise Induced Angina?', ['Yes', 'No'])
        oldpeak = st.slider('ST Depression Induced by Exercise', 0.0, 6.2, 3.2)
        slope = st.selectbox('Slope of Peak Exercise ST Segment', ['Upsloping', 'Flat', 'Downsloping'])
        ca = st.select_slider('Number of Major Vessels Colored by Fluoroscopy', options=[0, 1, 2, 3])
        thal = st.selectbox('Thalassemia', ['Normal', 'Fixed Defect', 'Reversible Defect'])

    with content_col:
        # Prepare user input for the model with exact feature names as used in training
        # Here, you map categorical inputs to numeric values based on your model's requirements
        sex_map = {'Male': 1, 'Female': 0}
        cp_map = {'Typical Angina': 0, 'Atypical Angina': 1, 'Non-anginal Pain': 2, 'Asymptomatic': 3}
        fbs_map = {'Yes': 1, 'No': 0}
        restecg_map = {'Normal': 0, 'ST-T wave abnormality': 1, 'Probable left ventricular hypertrophy': 2}
        exang_map = {'Yes': 1, 'No': 0}
        slope_map = {'Upsloping': 0, 'Flat': 1, 'Downsloping': 2}
        thal_map = {'Normal': 0, 'Fixed Defect': 1, 'Reversible Defect': 2}

        user_input_df = pd.DataFrame({
            'age': [age],
            'sex': [sex_map[sex]],
            'cp': [cp_map[cp]],
            'trestbps': [trestbps],
            'chol': [chol],
            'fbs': [fbs_map[fbs]],
            'restecg': [restecg_map[restecg]],
            'thalach': [thalach],
            'exang': [exang_map[exang]],
            'oldpeak': [oldpeak],
            'slope': [slope_map[slope]],
            'ca': [ca],
            'thal': [thal_map[thal]]
        })

        # Predict heart disease probability
        heart_disease_probability = predict_heart_disease(user_input_df)
        severe_heart_disease_probability = heart_disease_probability * 0.7  # Example adjustment
        feedback = provide_feedback(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Displaying the prediction results and health metrics in separate containers
        with st.container():
            st.subheader('👤 Your Health Metrics:')
            st.json({
                "Age": age,
                "Cholesterol": f"{chol} mg/dL - {feedback['Cholesterol']}",
                "Max Heart Rate Achieved": f"{thalach} bpm - {feedback['Max Heart Rate']}",
                "Resting Blood Pressure": f"{trestbps} mmHg - {feedback['Resting Blood Pressure']}"
            })

        with st.container():
            st.subheader('🔍 Heart Disease Assessment')
            # Display each metric in its own row
            st.metric("Probability of Heart Disease", f"{heart_disease_probability * 100:.2f}%")
            st.metric("Probability of Severe Heart Disease", f"{severe_heart_disease_probability * 100:.2f}%")
            status_color = "🔴" if heart_disease_probability > 0.5 else "🟢"
            st.metric("Diagnosis", f"{status_color} {'Heart Disease' if heart_disease_probability > 0.5 else 'No Heart Disease'}")

        with st.container():
            if heart_disease_probability > 0.5:
                st.subheader('🩺 Precautions and Recommendations')
                st.markdown("""
                - *Maintain a healthy diet:* Reduce saturated fats, salt, and added sugars.
                - *Regular physical activity:* Aim for at least 150 minutes of moderate-intensity exercise weekly.
                - *Quit smoking:* Smoking increases heart disease risk significantly.
                - *Manage stress:* Practice relaxation techniques such as yoga or meditation.
            """)
            else:
                st.markdown("you are fines")
        with st.container():
            st.subheader('👩‍⚕ Find Related Specialists')
            if heart_disease_probability > 0.5:
                st.markdown("""
                Based on your inputs, you may want to consult:
                - *Cardiologist:* For comprehensive heart health evaluation.
                - *Nutritionist:* To optimize your diet for heart health.
                - *Endocrinologist:* If diabetes management is also a concern.
            """)
            else:
                st.markdown("maintain healthy lifestyle")

    st.subheader('📈 Visual Health Analysis')
    plot_health_metrics(chol, thalach, trestbps)

# Run the app
if _name_ == '_main_':
    heart_disease_app()