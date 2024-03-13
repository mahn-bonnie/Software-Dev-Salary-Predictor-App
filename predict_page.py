import streamlit as st
import pickle
import numpy as np


def load_model():
    try:
        with open('saved_steps.pkl', 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        print("Error: File 'saved_steps.pkl' not found.")
        return None
    except pickle.UnpicklingError as e:
        print(f"Error: Failed to unpickle data - {e}")
        return None
    except Exception as e:
        print(f"Error: An unexpected error occurred - {e}")
        return None

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

# Define CSS styles for the title
title_style = """
    color: #007bff;
    font-size: 36px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 30px;
"""

def show_predict_page():
    # Render the styled title using markdown
    st.markdown("<h1 style='{}'>SOFTWARE DEVELOPER SALARY PREDICTION</h1>".format(title_style), unsafe_allow_html=True)

    # st.title("SOFTWARE DEVELOPER SALARY PREDICTION")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is $ {salary[0]:.2f}")
