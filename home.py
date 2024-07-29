import streamlit as st
def main():
    
    # CSS for styling
    st.markdown(
        """
        <style>
        .main-container {
            padding: 2rem;
            text-align: center;
        }
        .header {
            font-size: 3rem;
            color: #02ab21; /* Green color */
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 2rem;
            color: #333;
            margin-bottom: 1.5rem;
        }
        .tool {
            margin-bottom: 3rem;
            text-align: left;
            max-width: 800px;
            margin: auto;
        }
        .tool h2 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        .tool p {
            font-size: 1.2rem;
            line-height: 1.6;
            color: #666;
        }
        .tool img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .tool a {
            color: #02ab21; /* Green color */
            text-decoration: none;
            font-weight: bold;
        }
        .field-description {
            margin-top: 1rem;
            font-size: 1rem;
            color: #888;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    st.markdown('<h1 class="header">Health Prediction Tools</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predictive tools for diabetes, heart disease, and breast cancer detection</p>', unsafe_allow_html=True)

    # Diabetes Prediction Tool
    st.markdown('<div class="tool">', unsafe_allow_html=True)
    st.markdown('<img src="https://dynamicpts.com/wp-content/uploads/2022/07/diab.jpg" alt="Diabetes Prediction" />', unsafe_allow_html=True)
    st.markdown('<h2>Diabetes Prediction</h2>', unsafe_allow_html=True)
    st.markdown('<p>Advanced AI model to predict diabetes risk based on health metrics. Learn more about <a href="https://example.com/diabetes" target="_blank">diabetes prediction</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p class="field-description"><strong>Input Fields:</strong> Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Heart Disease Prediction Tool
    st.markdown('<div class="tool">', unsafe_allow_html=True)
    st.markdown('<img src="https://th.bing.com/th/id/R.ce5e9d964f027f5859ca8613f33b2e1c?rik=%2bI%2bJM95axu0bZg&riu=http%3a%2f%2fmedia.clinicaladvisor.com%2fimages%2f2017%2f03%2f29%2fheartillustrationts51811362_1191108.jpg&ehk=5j%2bzmEbCqhDShHbQaCiXdMZby7zNV7EghLdnSeu%2fGNM%3d&risl=&pid=ImgRaw&r=0" alt="Heart Disease Prediction" />', unsafe_allow_html=True)
    st.markdown('<h2>Heart Disease Prediction</h2>', unsafe_allow_html=True)
    st.markdown('<p>Predictive analytics tool for assessing heart disease probability. Explore more about <a href="https://example.com/heart-disease" target="_blank">heart disease prediction</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p class="field-description"><strong>Input Fields:</strong> Age, Gender, Chest Pain Type, Resting Blood Pressure, Serum Cholesterol, Fasting Blood Sugar, Resting ECG, Maximum Heart Rate, Exercise Induced Angina, ST Depression, Slope of ST Segment, Number of Major Vessels, Thalassemia</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Breast Cancer Detection Tool
    st.markdown('<div class="tool">', unsafe_allow_html=True)
    st.markdown('<img src="https://wallpapercave.com/wp/wp3755931.jpg" alt="Breast Cancer Detection" />', unsafe_allow_html=True)
    st.markdown('<h2>Breast Cancer Detection</h2>', unsafe_allow_html=True)
    st.markdown('<p>AI-powered tool for early detection of breast cancer using deep learning. Read more about <a href="https://example.com/breast-cancer" target="_blank">breast cancer detection</a>.</p>', unsafe_allow_html=True)
    st.markdown('<p class="field-description"><strong>Input Fields:</strong> Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave Points, Symmetry, Fractal Dimension</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if _name_ == "_main_":
    main()