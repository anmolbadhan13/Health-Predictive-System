import streamlit as st
from dotenv import load_dotenv
from streamlit_option_menu import option_menu

# Load environment variables (if necessary)
load_dotenv()

# Importing different app modules
import home
import diabetes_project
import heart2
import breast_cancer_project

# Set Streamlit page configuration
st.set_page_config(
    page_title="Pondering"
)

# Define styles for the sidebar with white background and adjusted colors
styles = {
    "container": {"padding": "10px 15px", "background-color": "#ffffff"},  # White background color
    "icon": {"color": "#333", "font-size": "20px"},  # Smaller icon and darker color
    "nav-link": {
        "color": "#333",  # Smaller font and darker color
        "font-size": "18px",
        "text-align": "left",
        "margin": "0px",
        "--hover-color": "#ffffff"  # Adjusted hover color to white
    },
    "nav-link-selected": {"background-color": "#ffffff", "color": "#02ab21"}  # Selected link background color and text color
}

# Adding multiple apps to the menu using MultiApp class
class MultiApp:
    def _init_(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        # Display the sidebar with options
        with st.sidebar:
            # Use option_menu for navigation
            app = option_menu(
                menu_title='Pondering',
                options=['Home', 'Diabetes Prediction', 'Heart Disease Prediction', 'Breast Cancer Detection'],
                default_index=0,  # Adjust default index based on your preference
                styles=styles
            )

        # Run the selected app based on the menu choice
        for item in self.apps:
            if item["title"] == app:
                item["function"]()

# Initialize the MultiApp class
app = MultiApp()

# Add different apps to the menu
app.add_app("Home", home.main)
app.add_app("Diabetes Prediction", diabetes_project.diabetes_app)
app.add_app("Heart Disease Prediction", heart2.heart_disease_app)
app.add_app("Breast Cancer Detection", breast_cancer_project.breast_app)

# Run the app
if _name_ == "_main_":
    app.run()
