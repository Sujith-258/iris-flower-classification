
# Import required libraries

import pickle
from sklearn.datasets import load_iris

# Step 1: Load saved model

with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Step 2: Load flower names

iris = load_iris()

# Step 3: Enter new flower measurements

new_flower = [[5.1, 3.5, 1.4, 0.2]]

# Step 4: Predict flower species

prediction = model.predict(new_flower)

# Step 5: Display prediction

flower_name = iris.target_names[prediction[0]]

print("Predicted Flower:", flower_name)
