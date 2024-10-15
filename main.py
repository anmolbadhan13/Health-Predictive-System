import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import tensorflow as tf  # Import TensorFlow
from PIL import Image

# Load the trained models
diabetes_model = pickle.load(open('diabetes_model_logistic.sav', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))  # Adjust this path if needed
heart_model = pickle.load(open('heart_model.sav', 'rb'))
heart_scaler = pickle.load(open('heart_scaler.pkl', 'rb'))  # Adjust this path if needed
# Load the breast cancer model and scaler
breast_cancer_model = tf.keras.models.load_model('breast_cancer_model_1.h5')  # Load the TensorFlow model
breast_cancer_scaler = pickle.load(open('breast_cancer_scaler.pkl', 'rb'))  # Directly load the scaler with pickle


# Sidebar for navigation
# Sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Home', 'Diabetes Prediction',
                            'Heart Disease Prediction', 'Breast Cancer Prediction'],
                           icons=['house', 'droplet', 'heart', 'gender-female'],  # Use 'female' as the icon for breast cancer
                           default_index=0)

# Sample Data
data = {
    "Disease": ["Diabetes", "Heart Disease", "Breast Cancer"],
    "Global Cases (2024)": [537, 523, 185],  # Replace with actual data
    "Projected Cases (2030)": [641, 699, 250]  # Replace with actual data
}

# Create a DataFrame
df = pd.DataFrame(data)
# Custom CSS for background color and logo
st.markdown(
    """
    <style>
    body {
        background-color: #f9f9f9;  /* Light gray background */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Green */
        color: white;
        border: none;
        padding: 15px 32px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# Home page content
if selected == 'Home':
    st.title("Welcome to the Multiple Disease Prediction System ")

    # Add a logo at the top of the page
    st.image("logo2.png", width=100)  # Update with the correct path to your logo image

    st.write(
        """
        This system is designed to help users predict the likelihood of developing certain diseases based on their input data. 
        It focuses on three major health conditions: diabetes, heart disease, and breast cancer.
        """
    )

    st.header("What You Will Learn")
    st.write(
        """
        - **Diabetes**: Diabetes is a chronic condition that occurs when the body cannot effectively use insulin. 
        It can lead to serious health complications if not managed properly.

        - **Heart Disease**: Heart disease refers to a range of conditions that affect the heart, including coronary artery disease, 
        heart rhythm problems, and heart defects. Risk factors include high blood pressure, high cholesterol, and obesity.

        - **Breast Cancer**: Breast cancer is a type of cancer that forms in the cells of the breasts. 
        Early detection through screening can significantly improve the chances of successful treatment.
        """
    )

    st.header("Why This System is Important")
    st.write(
        """
        Early detection and prevention can save lives. By using our predictive models, you can gain insights into your health 
        and take proactive steps to address potential risks.
        """
    )

    st.header("How to Use This System")
    st.write(
        """
        1. Navigate to the respective disease prediction pages from the sidebar.
        2. Enter your details as prompted.
        3. Click on the 'Predict' button to see the results.
        4. Consult with a healthcare professional for further guidance based on the results.
        """
    )

    # Create bar charts for disease statistics
    st.header("Global Health Statistics")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Bar chart for Global Cases
    ax[0].bar(df["Disease"], df["Global Cases (2024)"], color='lightblue')
    ax[0].set_title("Global Cases in 2024")
    ax[0].set_ylabel("Number of Cases (in millions)")

    # Bar chart for Projected Cases
    ax[1].bar(df["Disease"], df["Projected Cases (2030)"], color='lightcoral')
    ax[1].set_title("Projected Cases by 2030")
    ax[1].set_ylabel("Number of Cases (in millions)")

    # Display charts in Streamlit
    st.pyplot(fig)

    # Optional insights
    st.write("""
        The bar charts above illustrate the current and projected cases of diabetes, heart disease, and breast cancer.
        It is crucial to focus on prevention and awareness to combat these growing health challenges.
    """)


# Diabetes Prediction Page
elif selected == "Diabetes Prediction":
    st.subheader("Diabetes Prediction")

    # Input fields for diabetes prediction
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=0)
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, value=0)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=0.0)
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.0)
    age = st.number_input("Age", min_value=0, max_value=120, value=0)

    # Prepare the feature array for prediction
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardize the data
    std_data = diabetes_scaler.transform(input_data_reshaped)

    # Button to get prediction
    if st.button('Diabetes Test Result'):
        prediction = diabetes_model.predict(std_data)
        if prediction[0] == 0:
            st.markdown('<p style="color:green;">The person is not diabetic.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:red;">The person is diabetic.</p>', unsafe_allow_html=True)



# Heart Disease Prediction Page
elif selected == "Heart Disease Prediction":
    st.subheader("Heart Disease Prediction")

    # Input fields for heart disease prediction
    age = st.number_input("Age", min_value=0, max_value=120, value=0)
    sex = st.selectbox("Gender (0 = Male, 1 = Female)", options=[0, 1])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=200, value=0)
    chol = st.number_input("Cholesterol Level (mg/dl)", min_value=0, max_value=600, value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl? (0 = No, 1 = Yes)", options=[0, 1])
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=0, max_value=200, value=0)
    exang = st.selectbox("Exercise-Induced Angina? (0 = No, 1 = Yes)", options=[0, 1])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, value=0.0)
    slope = st.selectbox("Slope of the Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)",
                         options=[0, 1, 2])

    # Chest pain type one-hot encoding
    cp = st.selectbox("Chest Pain Type (0 = Typical, 1 = Atypical, 2 = Non-Anginal, 3 = Asymptomatic)",
                      options=[0, 1, 2, 3])

    # Resting electrocardiographic results one-hot encoding
    restecg = st.selectbox(
        "Resting Electrocardiographic Results (0 = Normal, 1 = ST-T Wave Abnormality, 2 = Left Ventricular Hypertrophy)",
        options=[0, 1, 2])

    # Thalassemia stress test result one-hot encoding
    thal = st.selectbox("Thalassemia (0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect)", options=[0, 1, 2])

    # Prepare the feature array for prediction
    input_data = [
        age,
        sex,
        trestbps,
        chol,
        fbs,
        thalach,
        exang,
        oldpeak,
        slope,
        0,  # ca (Assuming no specific input for ca is needed, you may set it to a default value)
        0,  # cp_1
        0,  # cp_2
        0,  # cp_3
        0,  # restecg_1
        0,  # restecg_2
        0,  # thal_1
        0,  # thal_2
        0  # thal_3 (if applicable, can be set based on thal value)
    ]

    # One-hot encoding logic based on user input
    if cp == 1:
        input_data[11] = 1  # cp_1
    elif cp == 2:
        input_data[12] = 1  # cp_2
    elif cp == 3:
        input_data[13] = 1  # cp_3

    if restecg == 1:
        input_data[14] = 1  # restecg_1
    elif restecg == 2:
        input_data[15] = 1  # restecg_2

    if thal == 1:
        input_data[16] = 1  # thal_1
    elif thal == 2:
        input_data[17] = 1  # thal_2
    elif thal == 3:
        input_data[18] = 1  # thal_3

    # Convert the input data to a numpy array
    input_array_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_array_as_numpy_array.reshape(1, -1)

    # Ensure the correct number of features
    if input_data_reshaped.shape[1] != 18:
        st.error("The input data does not have the expected number of features.")
    else:
        # Standardize the data using the robust scaler
        std_data = heart_scaler.transform(input_data_reshaped)

        # Button to get prediction
        if st.button('Heart Disease Test Result'):
            prediction = heart_model.predict(std_data)
            if prediction[0] == 0:
                st.markdown('<p style="color:green;">The person is not at risk for heart disease.</p>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:red;">The person is at risk for heart disease.</p>',
                            unsafe_allow_html=True)






# Breast Cancer Prediction Page
elif selected == "Breast Cancer Prediction":
    st.subheader("Breast Cancer Prediction")

    # Input fields for breast cancer prediction (30 inputs)
    mean_radius = st.number_input("Mean Radius", value=14.0)
    mean_texture = st.number_input("Mean Texture", value=7.0)
    mean_perimeter = st.number_input("Mean Perimeter", value=13.0)
    mean_area = st.number_input("Mean Area", value=25.0)
    mean_smoothness = st.number_input("Mean Smoothness", value=6.0)
    mean_compactness = st.number_input("Mean Compactness", value=16.0)
    mean_concavity = st.number_input("Mean Concavity", value=18.0)
    mean_concave_points = st.number_input("Mean Concave Points", value=10.0)
    mean_symmetry = st.number_input("Mean Symmetry", value=15.0)
    mean_fractal_dimension = st.number_input("Mean Fractal Dimension", value=15.0)
    radius_error = st.number_input("Radius Error", value=38.0)
    texture_error = st.number_input("Texture Error", value=20.0)
    perimeter_error = st.number_input("Perimeter Error", value=38.0)
    area_error = st.number_input("Area Error", value=65.0)
    smoothness_error = st.number_input("Smoothness Error", value=30.0)
    compactness_error = st.number_input("Compactness Error", value=28.0)
    concavity_error = st.number_input("Concavity Error", value=22.0)
    concave_points_error = st.number_input("Concave Points Error", value=19.0)
    symmetry_error = st.number_input("Symmetry Error", value=27.0)
    fractal_dimension_error = st.number_input("Fractal Dimension Error", value=28.0)

    # Additional features (if applicable)
    worst_radius = st.number_input("Worst Radius", value=17.0)
    worst_texture = st.number_input("Worst Texture", value=5.0)
    worst_perimeter = st.number_input("Worst Perimeter", value=15.0)
    worst_area = st.number_input("Worst Area", value=35.0)
    worst_smoothness = st.number_input("Worst Smoothness", value=7.0)
    worst_compactness = st.number_input("Worst Compactness", value=16.0)
    worst_concavity = st.number_input("Worst Concavity", value=12.0)
    worst_concave_points = st.number_input("Worst Concave Points", value=0.0)
    worst_symmetry = st.number_input("Worst Symmetry", value=23.0)
    worst_fractal_dimension = st.number_input("Worst Fractal Dimension", value=24.0)

    # Prepare the feature array for prediction
    input_data = (mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
                  mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
                  mean_fractal_dimension, radius_error, texture_error, perimeter_error,
                  area_error, smoothness_error, compactness_error, concavity_error,
                  concave_points_error, symmetry_error, fractal_dimension_error,
                  worst_radius, worst_texture, worst_perimeter, worst_area,
                  worst_smoothness, worst_compactness, worst_concavity, worst_concave_points,
                  worst_symmetry, worst_fractal_dimension)

    input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

    # Standardize the data
    input_data_std = breast_cancer_scaler.transform(input_data_as_numpy_array)

    # Button to get prediction
    if st.button('Breast Cancer Test Result'):
        prediction = breast_cancer_model.predict(input_data_std)
        prediction_label = np.argmax(prediction, axis=1)
        if prediction_label[0] == 0:
            st.markdown('<p style="color:red;">The tumor is Malignant.</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p style="color:green;">The tumor is Benign.</p>', unsafe_allow_html=True)



