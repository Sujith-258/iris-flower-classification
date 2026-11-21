
# Import required libraries

import pandas as pd
from sklearn.datasets import load_iris

# Step 1: Load dataset

iris = load_iris()

# Step 2: Create DataFrame

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Step 3: Create input features (X)

X = df

# Step 4: Create target variable (y)

y = iris.target

# Step 5: Display X

print("Input Features (X):")
print(X.head())

# Step 6: Display y

print("\nTarget Variable (y):")
print(y[:5])

# Step 7: Display shapes

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)


# Import train_test_split

from sklearn.model_selection import train_test_split

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Display training and testing shapes

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nTesting Target Shape:")
print(y_test.shape)
