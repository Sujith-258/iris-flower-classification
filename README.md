# Iris Flower Classification 🌸

A Machine Learning project that predicts the species of an Iris flower based on its sepal and petal measurements using Logistic Regression.

## 📌 Project Overview

The Iris Flower Classification project uses machine learning to classify flowers into three different species:

- Setosa
- Versicolor
- Virginica

The model takes four input measurements and predicts the most likely flower species.

## 🎯 Project Objective

The main objectives of this project are:

- Understand the fundamentals of Machine Learning.
- Explore and visualize a real-world dataset.
- Perform data preprocessing.
- Train a Logistic Regression classification model.
- Evaluate model performance.
- Build an interactive web application using Streamlit.

## 🛠️ Technologies Used

- **Python** – Programming language
- **NumPy** – Numerical operations
- **Pandas** – Data manipulation and analysis
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical visualization
- **Scikit-learn** – Machine Learning model and evaluation
- **Streamlit** – Interactive web application
- **Pickle** – Saving and loading the trained model
- **Git & GitHub** – Version control and project hosting

## 📂 Project Structure

```text
Iris-Flower-Classification/
│
├── app.py                # Streamlit web application
├── train.py              # Model training and evaluation
├── preprocessing.py      # Data preprocessing
├── explore_data.py       # Exploratory Data Analysis
├── predict.py            # Flower species prediction
├── check.py              # Library installation verification
├── iris_model.pkl        # Saved trained ML model
├── requirements.txt      # Project dependencies
├── .gitignore            # Git ignored files
└── README.md             # Project documentation
```

## 📊 Dataset Information

This project uses the built-in Iris dataset from Scikit-learn.

- Total records: 150
- Total features: 4
- Target classes: 3
- Records per class: 50

### Input Features

1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

### Target Classes

- Setosa
- Versicolor
- Virginica

## ⚙️ Machine Learning Workflow

1. Load the Iris dataset.
2. Perform Exploratory Data Analysis (EDA).
3. Visualize the dataset using graphs.
4. Separate input features and target labels.
5. Split the dataset into training and testing sets (80:20).
6. Train the Logistic Regression model.
7. Evaluate the model using accuracy, confusion matrix, and classification report.
8. Save the trained model using Pickle.
9. Build an interactive prediction application using Streamlit.

## 📈 Model Performance

**Algorithm:** Logistic Regression

**Training and Testing Split:** 80% Training / 20% Testing

**Test Accuracy:** 96.67%

The model correctly predicted 29 out of 30 test samples.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Sujith-258/iris-flower-classification.git
```

### 2. Navigate to the project directory

```bash
cd iris-flower-classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

For Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 5. Install the required libraries

```bash
pip install -r requirements.txt
```

## ▶️ Run the Project

### Run the Machine Learning prediction script

```bash
python predict.py
```

### Run the Streamlit web application

```bash
python -m streamlit run app.py
```

The application will open in your browser. Enter the four flower measurements and click the prediction button to see the predicted species and estimated class probabilities.

## 💡 Key Learnings

Through this project, I gained practical experience in:

- Python programming for Machine Learning.
- Data cleaning and preprocessing.
- Exploratory Data Analysis.
- Data visualization.
- Logistic Regression classification.
- Model evaluation.
- Saving and loading trained ML models.
- Building interactive ML applications with Streamlit.

## 👨‍💻 Author

**Sujith D**

GitHub: [Sujith-258](https://github.com/Sujith-258)

---

⭐ If you find this project useful, consider giving it a star!
