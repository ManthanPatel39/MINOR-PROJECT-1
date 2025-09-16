import streamlit as st
import pickle
import numpy as np

# Load model and artifacts
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('locations.pkl', 'rb') as f:
        locations = pickle.load(f)
    
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    
    location_columns = [col for col in columns if col.startswith('location_')]
except Exception as e:
    st.error(f"Error loading model or artifacts: {e}")
    model = None
    locations = []
    columns = []

# Streamlit app UI
st.title("🏡 House Price Prediction App")

if model is None:
    st.warning("Model or artifacts not loaded. Cannot make predictions.")
else:
    # Inputs
    total_sqft = st.number_input("Total Square Feet", min_value=300, max_value=10000, step=50)
    bhk = st.slider("Number of Bedrooms (BHK)", 1, 10, 3)
    bath = st.slider("Number of Bathrooms", 1, 10, 2)
    balcony = st.slider("Number of Balconies", 0, 5, 1)
    location = st.selectbox("Select Location", sorted(locations))

    if st.button("Predict Price"):
        try:
            # Prepare input array
            x = np.zeros(len(columns))
            x[0] = total_sqft
            x[1] = bath
            x[2] = bhk
            x[3] = balcony

            # Handle one-hot location
            loc_column_name = f'location_{location}'
            if loc_column_name in columns:
                loc_index = columns.index(loc_column_name)
                x[loc_index] = 1

            # Prediction
            predicted_price = model.predict([x])[0]
            predicted_price = max(0, predicted_price)  # avoid negative

            # Format output
            if predicted_price >= 100:
                formatted_price = f"{predicted_price/100:.2f} Cr"
            else:
                formatted_price = f"{predicted_price:.2f} Lakhs"

            st.success(f"Predicted Price: {formatted_price}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
