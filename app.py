import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# Load the trained model and artifacts
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('locations.pkl', 'rb') as f:
        locations = pickle.load(f)

    with open('columns.pkl', 'rb') as f:
        columns = pickle.load(f)

    location_columns = [col for col in columns if col.startswith('location_')]

    print("Model, locations, and columns loaded successfully.")
    print(f"Sample locations: {locations[:5]}")
    print(f"Total columns expected by model: {len(columns)}")

except FileNotFoundError:
    print("Error: model.pkl, locations.pkl, or columns.pkl not found.")
    model, locations, columns = None, [], []
except Exception as e:
    print(f"Error loading model or artifacts: {e}")
    model, locations, columns = None, [], []


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


@app.route('/predict', methods=['POST'])
def predict_home_price():
    """Receives house details from the frontend and returns a formatted price prediction."""
    if model is None or not locations or not columns:
        return jsonify({'error': 'Model or necessary data not loaded. Cannot make predictions.'}), 500

    data = request.get_json()

    try:
        total_sqft = float(data['total_sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        balcony = int(data['balcony'])
        location = data['location']

        # Input vector
        x = np.zeros(len(columns))
        x[0] = total_sqft
        x[1] = bath
        x[2] = bhk
        x[3] = balcony

        # One-hot encoding for location
        try:
            loc_index = columns.index(f'location_{location}')
            x[loc_index] = 1
        except ValueError:
            print(f"Warning: Location '{location}' not found in model's known locations.")

        # Predict price
        predicted_price = model.predict([x])[0]
        predicted_price = max(0, predicted_price)  # prevent negatives

        # Format price
        if predicted_price >= 100:  # >= 100 Lakhs = 1 Cr
            formatted_price = f"₹{predicted_price/100:.2f} Cr"
        else:
            formatted_price = f"₹{predicted_price:.2f} Lakhs"

        return jsonify({
            'predicted_price': round(predicted_price, 2),
            'formatted_price': formatted_price
        })

    except KeyError as e:
        return jsonify({'error': f'Missing data: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data type: {e}'}), 400
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during prediction: {e}'}), 500


if __name__ == '__main__':
    # In production, use Gunicorn (Render will handle this automatically)
    app.run(debug=True)
