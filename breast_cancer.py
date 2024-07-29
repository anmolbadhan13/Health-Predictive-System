import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from tensorflow.keras.models import load_model
import pickle


# Function to load the scaler
def load_scaler():
    current_dir = os.path.dirname(_file_)
    scaler_path = os.path.join(current_dir, 'scalers2.pkl')

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)

    return scaler


# Function to add sidebar sliders
def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")

    example_max_values = {
        "radius_mean": 28.0,
        "texture_mean": 40.0,
        "perimeter_mean": 200.0,
        "area_mean": 2500.0,
        "smoothness_mean": 0.2,
        "compactness_mean": 1.2,
        "concavity_mean": 1.5,
        "concave points_mean": 0.4,
        "symmetry_mean": 0.5,
        "fractal_dimension_mean": 0.1,
        "radius_se": 3.0,
        "texture_se": 5.0,
        "perimeter_se": 20.0,
        "area_se": 400.0,
        "smoothness_se": 0.05,
        "compactness_se": 0.3,
        "concavity_se": 0.4,
        "concave points_se": 0.1,
        "symmetry_se": 0.1,
        "fractal_dimension_se": 0.03,
        "radius_worst": 40.0,
        "texture_worst": 50.0,
        "perimeter_worst": 300.0,
        "area_worst": 3000.0,
        "smoothness_worst": 0.3,
        "compactness_worst": 2.0,
        "concavity_worst": 2.5,
        "concave points_worst": 0.5,
        "symmetry_worst": 0.7,
        "fractal_dimension_worst": 0.2
    }

    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=0.0,
            max_value=example_max_values[key],
            value=example_max_values[key] / 2
        )

    return input_dict


# Function to scale input data using the loaded scaler
def get_scaled_values(input_dict, scaler):
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    scaled_array = scaler.transform(input_array)
    return dict(zip(input_dict.keys(), scaled_array.flatten()))


# Function to generate radar chart based on scaled input data
def get_radar_chart(input_data, scaler):
    input_data = get_scaled_values(input_data, scaler)

    categories = ['Radius', 'Texture', 'Perimeter',
                  'Area', 'Smoothness', 'Compactness',
                  'Concavity', 'Concave Points',
                  'Symmetry', 'Fractal Dimension']

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean Value'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst Value'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        autosize=True
    )

    return fig


# Function to load and predict using the model
def add_predictions(input_data, scaler):
    current_dir = os.path.dirname(_file_)
    model_path = os.path.join(current_dir, 'rajas2.h5')

    # Load the model
    model = load_model(model_path)

    # Scale input data
    scaled_input_data = get_scaled_values(input_data, scaler)
    scaled_input_array = np.array(list(scaled_input_data.values())).reshape(1, -1)

    # Make predictions
    prediction = model.predict(scaled_input_array)

    st.subheader("Cell Cluster Prediction")
    st.write("The Cell cluster is : ")

    if prediction[0][0] < 0.5:
        st.write("<span class='diag benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write("<span class='diag malicious'>Malicious</span>", unsafe_allow_html=True)

    st.write("Probability of being Benign: ", 1 - prediction[0][0])
    st.write("Probability of being Malicious: ", prediction[0][0])
    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.")


# Main function for the breast cancer detection app
def breast_app():
    st.title("Breast Cancer Prediction Application")

    # Load the scaler
    scaler = load_scaler()

    # Add sidebar sliders
    input_data = add_sidebar()

    # Display input measurements
    st.header("Input Measurements")
    st.write(pd.DataFrame([input_data]))

    # Display radar chart
    st.header("Radar Chart")
    fig = get_radar_chart(input_data, scaler)
    st.plotly_chart(fig)

    # Display prediction
    st.header("Prediction")
    add_predictions(input_data, scaler)


# Entry point for the application
if _name_ == "_main_":
     breast_app()
