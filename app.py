import streamlit as st
import numpy as np
import pickle
import joblib

# Load the model and label encoders
@st.cache_resource
def load_model():
    return joblib.load('visualization and modelling.pkl')

@st.cache_resource
def load_label_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        return pickle.load(file)

# Function to collect user input
def get_user_input():
    country = st.selectbox('Country', ['Kenya', 'Uganda', 'Tanzania', 'Rwanda'])
    location_type = st.radio('Location Type', ['Rural', 'Urban'])
    cellphone_access = st.selectbox('Cellphone Accessibility', ['Yes', 'No'])
    household_size = st.number_input('Household Size', min_value=1, max_value=50, value=1)
    age_of_respondent = st.number_input('Age of Respondent', min_value=18, max_value=100, value=25)
    gender_of_respondent = st.radio('Gender of Respondent', ['Male', 'Female'])
    marital_status = st.selectbox('Marital Status', ['Married/Living together', 'Divorced/separated', 'Widowed', 'Single/Never Married'])
    education_level = st.selectbox('Education Level', ['Secondary education', 'Primary education', 'No formal education', 'Vocational/Specialised training'])
    job_type = st.selectbox('Job Type', ['Dont Know/Refuse to answer', 'Farming and Fishing', 'Formally employed Government', 'Formally employed Private', 'Government Dependent', 'Informally employed', 'No Income', 'Other Income', 'Remittance Dependent', 'Self employed'])
    relationship_with_head = st.selectbox('Relationship with Head', ['Head of Household', 'Spouse', 'Child', 'Other relative', 'Parent'])

    return np.array([[ 
        label_encoders['country'].transform([country])[0],
        label_encoders['location_type'].transform([location_type])[0],
        label_encoders['cellphone_access'].transform([cellphone_access])[0],
        household_size,
        age_of_respondent,
        label_encoders['gender_of_respondent'].transform([gender_of_respondent])[0],
        label_encoders['marital_status'].transform([marital_status])[0],
        label_encoders['education_level'].transform([education_level])[0],
        label_encoders['job_type'].transform([job_type])[0],
        label_encoders['relationship_with_head'].transform([relationship_with_head])[0]
    ]])

# Main function for the Streamlit app
def main():
    model = load_model()
    label_encoders = load_label_encoders()

    st.title('Bank Account Ownership Prediction')
    st.write('Please provide the following details to predict whether an individual has a bank account.')

    input_data = get_user_input()

    # Check the shape of input data
    st.write('Input data shape:', input_data.shape)

    prediction_labels = {0: 'No Bank Account', 1: 'Bank Account'}

    if st.button('Predict'):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display prediction results
        prediction_result = prediction_labels[prediction[0]]
        st.write(f'Predicted outcome: {prediction_result}')
        st.write(f'Probability of having a bank account: {prediction_proba[0][1]:.2f}')
        st.write(f'Probability of not having a bank account: {prediction_proba[0][0]:.2f}')

        # Optionally display the raw prediction and probabilities for debugging
        st.write(f'Raw prediction: {prediction}')
        st.write(f'Prediction probabilities: {prediction_proba}')

if __name__ == '__main__':
    main()
