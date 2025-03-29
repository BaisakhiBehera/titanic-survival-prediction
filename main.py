#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd
import pickle
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'Titanic_train.csv'))

df


# In[3]:


df.head()
df.tail()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


# import matplotlib.pyplot as plt
# df.hist(bins=30, figsize=(15, 10), edgecolor='black')
# plt.suptitle('Histograms of All Numeric Features')
# plt.show()


# # In[7]:


# df.boxplot()
# plt.show()


# In[8]:


import seaborn as sns
sns.pairplot(df)
plt.show()


# In[9]:


import numpy as np

# 1. Identify missing values
missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[10]:


# 2. Decide on the imputation strategy
numerical_cols = df.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mean(), inplace=True)


# In[11]:


# Replace missing categorical values with mode
categorical_cols = df.select_dtypes(exclude=[np.number]).columns
for col in categorical_cols:
    if df[col].isnull().any():
        df[col].fillna(df[col].mode()[0], inplace=True)


# In[12]:


missing_values = df.isnull().sum()
print("Missing Values:\n", missing_values)


# In[13]:


#Encoding the categorical variables

# 1. Check the unique values of categorical columns
print("Unique values in 'Sex':", df['Sex'].unique())
print("Unique values in 'Embarked':", df['Embarked'].unique())


# In[14]:


# 2. One-Hot Encoding using pandas get_dummies()
df_encoded = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# In[15]:


# 3. Display the encoded dataset
print("Encoded Dataset:\n", df_encoded)


# In[16]:


# Model Building:
# Build a logistic regression model using appropriate libraries

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Example: Cleaning and preparing the dataset
# Drop non-numeric columns like 'Name', 'Ticket', 'Cabin' if not needed for modeling
df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)


# In[17]:


df.head()


# In[18]:


# Encode categorical variables (e.g., 'Sex', 'Embarked')
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


# In[19]:


df.tail()


# In[20]:


# Handle missing values if any (e.g., impute 'Age' with mean)
df['Age'].fillna(df['Age'].mean(), inplace=True)


# In[21]:


# Define features (X) and target variable (y)
X = df.drop('Survived', axis=1)
y = df['Survived']


# In[22]:


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[24]:


# Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[25]:


# Make predictions
y_pred = model.predict(X_test_scaled)


# In[26]:


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[27]:


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[28]:


# Train the model using training data

# Define features (X) and target variable (y)
X = df.drop(['Survived', 'PassengerId'], axis=1)
y = df['Survived']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[29]:


# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[30]:


# Build the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)


# In[31]:


# Make predictions on the training set (just for illustration)
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
print(X.columns)  # Check which columns were used during model training


# In[32]:
# Save the trained model and scaler using pickle
with open("logistic_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and Scaler Saved Successfully!")


# Evaluate the model on training set
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)


# In[33]:


# Evaluate the model on test set
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Accuracy:", test_accuracy)


# In[34]:


# Make predictions on the test set
y_pred = model.predict(X_test_scaled)


# In[35]:


# Model Evaluation

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc

# Make predictions on the testing set
y_pred = model.predict(X_test_scaled)

# Probability predictions for ROC curve
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # Probability predictions for ROC curve


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")


# In[36]:


# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# In[37]:


# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# In[38]:


# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[39]:


#Interpretation:	

coefficients = pd.DataFrame(data=model.coef_[0], index=X.columns, columns=['Coefficient'])
print(coefficients)


# In[40]:


# Extracting coefficients and their corresponding feature names
coefficients = model.coef_[0]
feature_names = X.columns

# Displaying coefficients with their corresponding feature names
coefficients_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients
})

# Sort the coefficients by their absolute values for better interpretation
coefficients_df['Abs_Coefficient'] = coefficients_df['Coefficient'].abs()
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

print("Coefficients:")
print(coefficients_df)


# In[41]:


# Deployment with Streamlit
#get_ipython().run_line_magic('pip', 'install streamlit')


# In[42]:

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

def load_model():
    with open("logistic_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

def preprocess_input(input_data, feature_columns):
    input_df = pd.DataFrame([input_data])
    
    # One-Hot Encoding
    input_df = pd.get_dummies(input_df, columns=['Sex', 'Embarked'], drop_first=True)
    
    # Ensure all required columns exist
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns
    
    # Reorder columns to match the trained model
    input_df = input_df[feature_columns]
    return input_df
def predict_survival_probability(model, scaler, input_data, feature_columns):
    input_df = pd.DataFrame([input_data])  # Convert input to DataFrame
    
    # Apply one-hot encoding to match training
    input_df = pd.get_dummies(input_df)  

    # Ensure all expected columns exist, fill missing ones with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0  # Add missing columns with default value 0

    # Reorder columns to match training data
    input_df = input_df[feature_columns]

    # Transform input data
    input_scaled = scaler.transform(input_df)

    # Predict probability
    prob = model.predict_proba(input_scaled)[:, 1]  # Get survival probability
    return prob


def main():
    model, scaler = load_model()
    
    # Feature columns used during training
    feature_columns = X.columns.tolist()
    st.title('Titanic Survival Prediction')
    st.markdown('Enter passenger details to predict survival probability:')
    
    # User input
    age = st.slider('Age', 0, 100, 30)
    sex = st.selectbox('Sex', ['male', 'female'])
    pclass = st.selectbox('Passenger Class', [1, 2, 3])
    fare = st.number_input('Fare', min_value=0.0, max_value=1000.0, value=50.0)
    parch = st.number_input('Parch (Parents/Children Aboard)', min_value=0, max_value=10, value=0)
    sibsp = st.number_input('SibSp (Siblings/Spouses Aboard)', min_value=0, max_value=10, value=0)
    embarked = st.selectbox('Embarked', ['C', 'Q', 'S'])
    
    input_data = {
        'Pclass': pclass,
        'Age': age,
        'Fare': fare,
        'Parch': parch,
        'SibSp': sibsp,
        'Sex': sex,
        'Embarked': embarked
    }
    
    if st.button('Predict'):
        prob = predict_survival_probability(model, scaler, input_data, feature_columns)
        st.success(f'Survival Probability: {prob[0]:.2f}')


if __name__ == '__main__':
    main()
