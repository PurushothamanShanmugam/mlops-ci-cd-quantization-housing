# predict.py

import joblib
import numpy as np

# Load the model
model = joblib.load("models/linear_model.joblib")

# Dummy test data (8 features like California housing)
test_sample = np.array([[8.3252, 41, 6.9841, 1.0238, 322, 2.5556, 37.88, -122.23]])

# Run prediction
prediction = model.predict(test_sample)
print(f"Prediction for test sample: {prediction}")
