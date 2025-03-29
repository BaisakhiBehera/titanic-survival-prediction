# titanic-survival-prediction
A Titanic survival prediction project using Logistic Regression. This project includes data preprocessing, model training, evaluation, and deployment using Streamlit. Users can input passenger details to predict their survival probability.
# Titanic Survival Prediction

This project is a machine learning-based web application that predicts the survival probability of Titanic passengers based on their details. The model is trained using logistic regression and deployed using Streamlit.

## Features
- Predicts survival probability based on user input
- Uses logistic regression for classification
- Implements data preprocessing (handling missing values, encoding categorical variables, and feature scaling)
- Interactive UI built with Streamlit

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit

## Dataset
The dataset used for training is the Titanic dataset from Kaggle, which includes passenger details such as age, gender, ticket class, fare, number of siblings/spouses aboard, and number of parents/children aboard.

## Installation
### Clone the repository:
```bash
git clone https://github.com/yourusername/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
### Train the Model
1. Run `main.py` to train the logistic regression model and save it:
```bash
python main.py
```
2. This will generate `logistic_model.pkl` and `scaler.pkl` files.

### Run the Web App
1. Start the Streamlit app:
```bash
streamlit run main.py
```
2. Open the provided link in your browser and enter passenger details to get survival predictions.


