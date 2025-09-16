import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained model and artifacts
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Assuming locations and columns are saved as separate pickle files or part of a larger artifact
    with open('locations.pkl', 'rb') as f:
        locations = pickle.load(f)
    
    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    
    # The first few columns are usually features like 'total_sqft', 'bath', 'bhk', 'balcony'
    # The rest are one-hot encoded locations.
    # We need to ensure 'location' is handled correctly in the input array.
    # The 'columns' list should contain all feature names in the order expected by the model.
    # Example: ['total_sqft', 'bath', 'bhk', 'balcony', 'location_Electronic City', 'location_Whitefield', ...]
    
    # Create a mapping for location columns for easier one-hot encoding
    # This assumes 'columns' contains 'location_...' entries
    location_columns = [col for col in columns if col.startswith('location_')]
    
    print("Model, locations, and columns loaded successfully.")
    print(f"Sample locations: {locations[:5]}")
    print(f"Total columns expected by model: {len(columns)}")
    
except FileNotFoundError:
    print("Error: model.pkl, locations.pkl, or columns.pkl not found.")
    print("Please ensure these files are in the same directory as app.py")
    model = None
    locations = []
    columns = []
except Exception as e:
    print(f"Error loading model or artifacts: {e}")
    model = None
    locations = []
    columns = []

@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    """Returns the list of unique locations for the dropdown."""
    if locations:
        return jsonify({'locations': sorted(locations)})
    return jsonify({'error': 'Locations not loaded'}), 500

@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    """
    Receives house details from the frontend and returns a price prediction.
    """
    if model is None or not locations or not columns:
        return jsonify({'error': 'Model or necessary data not loaded. Cannot make predictions.'}), 500

    data = request.get_json()

    try:
        total_sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        balcony = int(data['balcony'])
        location = data['location']

        # Create a zero array with the same length as model's expected input features
        # The order of features in 'columns' is crucial.
        x = np.zeros(len(columns))

        # Assign values to known features
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        x[3] = balcony # Assuming balcony is the 4th feature (index 3)

        # Handle one-hot encoding for location
        try:
            loc_index = columns.index(f'location_{location}')
            x[loc_index] = 1
        except ValueError:
            # If the location is new/unseen by the model, treat it as 'other' or a default.
            # For simplicity, we'll just let it remain 0 for all location columns if not found.
            # In a real-world scenario, you might have an 'other' category in your training data.
            print(f"Warning: Location '{location}' not found in model's known locations.")
            # If you have an 'other' column, uncomment and adjust this:
            # try:
            #     other_loc_index = columns.index('location_other')
            #     x[other_loc_index] = 1
            # except ValueError:
            #     pass # No 'other' category, so it remains all zeros for location.

        # Make prediction
        predicted_price = model.predict([x])[0]

        # Ensure price is not negative (can happen with linear models on unseen data)
        predicted_price = max(0, predicted_price)

        return jsonify({'predicted_price': predicted_price})

    except KeyError as e:
        return jsonify({'error': f'Missing data: {e}. Please provide all required fields.'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data type: {e}. Please ensure numbers are correct.'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during prediction: {e}'}), 500

if __name__ == '__main__':
    # This is for development. In production, use a WSGI server like Gunicorn.
    app.run(debug=True)