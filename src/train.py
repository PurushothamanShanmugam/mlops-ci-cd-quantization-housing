# train.py

from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

# Load dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Save the model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_model.joblib")

print("Model trained and saved as models/linear_model.joblib")
