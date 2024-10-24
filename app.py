import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv('financecleaned_data.csv')  # Your cleaned dataset

# Encode categorical columns using LabelEncoder
def label_encode_columns(df):
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':  # Only encode object (categorical) columns
            df[col] = le.fit_transform(df[col])
    return df

# Normalize the data using MinMaxScaler
def normalize_data(df, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler()
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        return df_scaled, scaler
    else:
        df_scaled = pd.DataFrame(scaler.transform(df), columns=df.columns)
        return df_scaled

# Streamlit app starts here
st.title("Random Forest Classifier Deployment")

# Load data
df = load_data()

# Encode categorical columns using LabelEncoder
df_encoded = label_encode_columns(df)

# Target variable (replace with actual column name)
target_variable = "bank_account"  # Adjust if necessary

# Define features and target variable
X = df_encoded.drop(['bank_account', 'uniqueid', 'country', 'year'], axis=1)  # Dropping unnecessary columns
y = df_encoded['bank_account']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the training and test data
X_train_scaled, scaler = normalize_data(X_train)
X_test_scaled = normalize_data(X_test, scaler=scaler)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

# User input form for new data
st.header("Input New Data for Prediction")
input_data = {}
for col in X.columns:
    input_data[col] = st.number_input(f"Enter {col}", value=float(X_train[col].mean()))  # Use mean as default value

# Convert user input into dataframe
input_df = pd.DataFrame([input_data])

# Normalize input data using the scaler from training data
normalized_input = normalize_data(input_df, scaler=scaler)

# Predict with the model
if st.button("Predict"):
    prediction = rf.predict(normalized_input)
    st.write(f"The predicted class is: {prediction[0]}")

# Optionally, show model accuracy
if st.checkbox("Show model accuracy"):
    accuracy = rf.score(X_test_scaled, y_test)
    st.write(f"Model accuracy: {accuracy * 100:.2f}%")
