# quantize.py

import joblib
import numpy as np
import os
import torch
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

# Load dataset
data = fetch_california_housing()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Load sklearn model
model = joblib.load("models/linear_model.joblib")

# Extract weights
weights = {
    "coef": model.coef_,
    "intercept": model.intercept_
}

# Save unquantized parameters
os.makedirs("quant", exist_ok=True)
joblib.dump(weights, "quant/unquant_params.joblib")

# Manual quantization (min-max scale to uint8)
def quantize_param(param):
    min_val = param.min()
    max_val = param.max()
    scale = (max_val - min_val) / 255
    quantized = ((param - min_val) / scale).astype(np.uint8)
    return quantized, scale, min_val

q_coef, scale_c, min_c = quantize_param(model.coef_)
q_intercept, scale_i, min_i = quantize_param(np.array([model.intercept_]))

# Save quantized params
quant_params = {
    "coef": q_coef,
    "intercept": q_intercept,
    "scale_coef": scale_c,
    "scale_intercept": scale_i,
    "min_coef": min_c,
    "min_intercept": min_i
}
joblib.dump(quant_params, "quant/quant_params.joblib")

# Dequantize
coef_deq = q_coef.astype(np.float32) * scale_c + min_c
intercept_deq = q_intercept.astype(np.float32)[0] * scale_i + min_i

# PyTorch inference
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
coef_tensor = torch.tensor(coef_deq, dtype=torch.float32)
intercept_tensor = torch.tensor(intercept_deq, dtype=torch.float32)

preds = X_test_tensor @ coef_tensor + intercept_tensor
r2_quant = r2_score(y_test, preds.detach().numpy())
r2_original = model.score(X_test, y_test)

print(f"R² Score - Original: {r2_original:.4f}")
print(f"R² Score - Quantized: {r2_quant:.4f}")

print("Unquantized model size:", os.path.getsize("quant/unquant_params.joblib") / 1024, "KB")
print("Quantized model size:", os.path.getsize("quant/quant_params.joblib") / 1024, "KB")
