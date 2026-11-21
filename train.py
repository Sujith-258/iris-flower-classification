
# Import required libraries

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 1: Load dataset

iris = load_iris()

X = iris.data
y = iris.target

# Step 2: Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 3: Create Logistic Regression model

model = LogisticRegression(max_iter=200)

# Step 4: Train the model

model.fit(X_train, y_train)

print("Model training completed!")

# Step 5: Predict using testing data

y_pred = model.predict(X_test)

# Step 6: Calculate accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")


# Step 7: Create Confusion Matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Step 8: Classification Report

print("\nClassification Report:")

print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# Step 9: Save trained model

import pickle

with open("iris_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel saved successfully!")
