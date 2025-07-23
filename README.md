# MLOps Assignment 3 - CI/CD Pipeline with Manual Quantization

## Author
Purushothaman Shanmugam

## Overview
This project demonstrates an end-to-end MLOps pipeline involving:
- Linear Regression with Scikit-learn
- PyTorch-based inference with manually quantized weights
- Docker containerization
- GitHub Actions-based CI/CD pipeline
- Manual INT8 quantization of model weights

---

## Branches

| Branch         | Purpose                               |
|----------------|---------------------------------------|
| `main`         | Base setup with .gitignore, README    |
| `dev`          | Model training script (`train.py`)    |
| `docker_ci`    | Docker + CI/CD pipeline via GitHub Actions |
| `quantization` | Manual quantization & PyTorch inference |

---

## Comparison Table

| Metric        | Original Model       | Quantized Model       |
|---------------|----------------------|------------------------|
| R² Score      | `<fill here>`        | `<fill here>`          |
| Model Size    | `<xx KB>`            | `<xx KB>`              |

---

## DockerHub
📤 Public image pushed: `https://hub.docker.com/r/<your-dockerhub-username>/housing-model`

## GitHub Repo
🔗 [https://github.com/PurushothamanShanmugam/mlops-ci-cd-quantization-housing](https://github.com/PurushothamanShanmugam/mlops-ci-cd-quantization-housing)

---

## How to Run Locally

```bash
# Train model
python src/train.py

# Run prediction
python src/predict.py

# Quantize model
python src/quantize.py
