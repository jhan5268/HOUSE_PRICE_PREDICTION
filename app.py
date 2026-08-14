import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Title and introduction
st.title("🏠 Hyderabad House Price Predictor")
st.write("Input property details below to estimate market value using Machine Learning.")

# Load the user-uploaded dataset to train the live interactive model
@st.cache_data
def load_and_train():
    df = pd.read_csv('data/hyd_houses_dataset.csv')
    core_features = ['Area', 'No. of Bedrooms', 'Resale', 'MaintenanceStaff', 'Gymnasium', 'SwimmingPool']
    df = df.dropna(subset=core_features + ['Price'])
    
    X = df[core_features]
    y = df['Price']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

try:
    model, scaler = load_and_train()

    # User Input UI Controls
    area = st.number_input("Property Area (in Sq. Ft.)", min_value=300, max_value=10000, value=1200)
    bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=6, value=2)
    resale = st.selectbox("Is it a Resale Property?", options=[("No", 0), ("Yes", 1)])[1]
    staff = st.checkbox("Maintenance Staff Available?")
    gym = st.checkbox("Gymnasium Included?")
    pool = st.checkbox("Swimming Pool Included?")

    # Calculation Button
    if st.button("Predict House Price"):
        user_features = np.array([[area, bedrooms, resale, int(staff), int(gym), int(pool)]])
        user_features_scaled = scaler.transform(user_features)
        
        predicted_price = model.predict(user_features_scaled)[0]
        
        st.success(f"💰 Estimated Price: ₹{predicted_price:,.2f}")

except Exception as e:
    st.error("Please ensure 'data/hyd_houses_dataset.csv' is properly configured.")
