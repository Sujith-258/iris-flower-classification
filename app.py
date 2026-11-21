
import streamlit as st
import pickle
from sklearn.datasets import load_iris

# Load trained model

with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load flower names

iris = load_iris()

# Application title

st.title("Iris Flower Classification")

st.write("Enter flower measurements to predict its species.")

# User inputs

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=4.0,
    max_value=8.0,
    value=5.1,
    step=0.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=1.5,
    max_value=5.0,
    value=3.5,
    step=0.1
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.5,
    max_value=7.5,
    value=1.4,
    step=0.1
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.1,
    max_value=3.0,
    value=0.2,
    step=0.1
)

# Prediction button

if st.button("Predict Flower"):

    # Prepare input data

    input_data = [[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]]

    # Predict flower species

    prediction = model.predict(input_data)

    flower_name = iris.target_names[prediction[0]]

    st.success(f"Predicted Flower: {flower_name}")

    # Calculate prediction probabilities

    probabilities = model.predict_proba(input_data)

    # Display probabilities

    st.subheader("Prediction Probabilities")

    for i, species in enumerate(iris.target_names):

        probability = probabilities[0][i] * 100

        st.write(f"{species}: {probability:.2f}%")

        st.progress(float(probabilities[0][i]))
