import streamlit as st
import pickle
import pandas as pd

# Load the machine learning model and scalers
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('min_max_scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('robust_scaler.pkl', 'rb') as file:
    robust_scaler = pickle.load(file)

# Streamlit App
st.title('Heart Disease Prediction App')

# Collect user input
user_input = []

st.subheader('User Input:')

# Numerical columns
NUMERICAL_COLS = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak']
for col in NUMERICAL_COLS:
    val = st.number_input(col, value=0.0)
    user_input.append(val)

# Categorical columns with dropdowns
CATEGORICAL_COLS = {
    'sex': ['Male', 'Female'],
    'cp': ['Asymptomatic', 'Atypical Angina', 'Non-Anginal', 'Typical Angina'],
    'fbs': ['False', 'True'],
    'restecg': ['LV Hypertrophy', 'Normal', 'ST-T Abnormality'],
    'exang': ['False', 'True']
}

for col, options in CATEGORICAL_COLS.items():
    user_input.append(st.selectbox(f'{col}', options))

# Convert the user input into a DataFrame
user_df = pd.DataFrame([user_input], columns=NUMERICAL_COLS + list(CATEGORICAL_COLS.keys()))
categorical_columns = list(CATEGORICAL_COLS.keys())
categorical_data = user_df[categorical_columns]
st.write(categorical_data)
# One-hot encoding for categorical variables
user_df_encoded = pd.get_dummies(categorical_data)
st.write(user_df_encoded)
expected_columns = model.feature_names_in_  # Obtain this from the trained model
user_df_encoded = user_df_encoded.reindex(columns=expected_columns,fill_value=0)
# Apply RobustScaler for handling outliers
user_df_encoded[NUMERICAL_COLS] = robust_scaler.transform(user_df_encoded[NUMERICAL_COLS])

# Feature scaling using Min-Max scaling
user_df_encoded[NUMERICAL_COLS] = scaler.transform(user_df_encoded[NUMERICAL_COLS])

# Display user input DataFrame
st.subheader('Processed User Input:')
st.write(user_df_encoded)

# Make predictions
prediction = model.predict(user_df_encoded)

# Display predictions
st.subheader('Prediction:')
st.write(f"The predicted class is: {prediction[0]}")

