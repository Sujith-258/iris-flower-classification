
# Import required libraries

import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns


# Step 1: Load the Iris dataset

iris = load_iris()

# Step 2: Convert dataset into DataFrame

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# Step 3: Add target column

df["species"] = iris.target

# Step 4: Display first five records

print(df.head())

# Step 5: Check dataset shape

print("\nDataset Shape:")
print(df.shape)

# Step 6: Check dataset information

print("\nDataset Information:")
df.info()

# Step 7: Statistical summary

print("\nStatistical Summary:")
print(df.describe())

# Step 8: Check missing values

print("\nMissing Values:")
print(df.isnull().sum())

# Step 9: Check species distribution

print("\nSpecies Distribution:")
print(df["species"].value_counts())

# Convert species numbers into flower names

df["species_name"] = df["species"].map({
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
})

print("\nUpdated Dataset:")
print(df.head())

# Create scatter plot

sns.scatterplot(
    data=df,
    x="petal length (cm)",
    y="petal width (cm)",
    hue="species_name"
)

# Add graph title

plt.title("Iris Flower Classification")

# Add axis labels

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")

# Display graph

plt.show()

# Step 2: Create Box Plot

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="species_name",
    y="petal length (cm)"
)

plt.title("Petal Length Distribution")
plt.xlabel("Flower Species")
plt.ylabel("Petal Length (cm)")

plt.show()

print("Box Plot displayed successfully!")



# Step 3: Create Pair Plot

sns.pairplot(
    df,
    hue="species_name"
)

plt.show()

print("Pair Plot displayed successfully!")
