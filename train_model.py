import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('MultipleFiles/Bengaluru_House_Data.csv')

# Data Cleaning and Preprocessing

# Drop columns not needed for prediction
df = df.drop(['area_type', 'availability', 'society'], axis=1)

# Drop rows with missing values
df = df.dropna()

# Convert 'size' to 'bhk' (e.g., '2 BHK' -> 2)
def extract_bhk(x):
    try:
        return int(x.split(' ')[0])
    except:
        return None

df['bhk'] = df['size'].apply(extract_bhk)
df = df.dropna(subset=['bhk'])

# Convert 'total_sqft' to numeric (handle ranges like '2100 - 2850')
def convert_sqft(x):
    try:
        if '-' in x:
            tokens = x.split('-')
            return (float(tokens[0].strip()) + float(tokens[1].strip())) / 2
        return float(x)
    except:
        return None

df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
df = df.dropna(subset=['total_sqft'])

# Bathrooms and Balcony columns: convert to int
df['bath'] = df['bath'].astype(int)
df['balcony'] = df['balcony'].astype(int)

# Clean location strings
df['location'] = df['location'].str.strip()

# Group locations with less than 10 occurrences as 'other'
location_counts = df['location'].value_counts()
locations_less_than_10 = location_counts[location_counts <= 10].index
df['location'] = df['location'].apply(lambda x: 'other' if x in locations_less_than_10 else x)

# One-hot encode location
dummies = pd.get_dummies(df['location'], prefix='location')

# Prepare final dataframe for modeling
df_model = pd.concat([df.drop(['location', 'size'], axis=1), dummies], axis=1)

# Features and target
X = df_model.drop('price', axis=1)
y = df_model['price']

# Save locations list for frontend dropdown (unique locations)
locations = sorted(df['location'].unique())

# Save columns list for model input order
model_columns = X.columns.tolist()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error on test set: {rmse:.2f} Lakhs")

# Save model and artifacts
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('locations.pkl', 'wb') as f:
    pickle.dump(locations, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(model_columns, f)

print("Model and artifacts saved successfully!")