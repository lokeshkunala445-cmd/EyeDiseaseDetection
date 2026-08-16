# EyeCare AI - Eye Disease Detection

This project is a web-based eye disease detection system built using Python, Flask and deep learning. The system takes an eye/retinal image as input and predicts the possible eye disease category.

## What the project does

The user can upload an eye image through the web application. The trained deep learning model analyzes the image and shows:

- Predicted eye disease
- Prediction confidence
- Uploaded image

The model has been trained to classify the following 5 categories:

1. Diabetic Retinopathy
2. Glaucoma
3. Healthy
4. Myopia
5. Retinitis Pigmentosa

## Technologies Used

- Python
- Flask
- TensorFlow / Keras
- NumPy
- Pillow
- HTML
- CSS
- MobileNetV2

## Running the Project

First, create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python app.py
```

Then open this address in the browser:

http://127.0.0.1:5000

## Model

The final trained model used by the application is:

`eye_disease_model_FINAL.keras`

The uploaded image is resized to 224 × 224 pixels and normalized before it is passed to the model for prediction.

## Project Structure

```text
EyeDiseaseDetection_GitHub/
│
├── app.py
├── eye_disease_model_FINAL.keras
├── requirements.txt
├── README.md
│
└── templates/
    └── index.html
```

## Disclaimer

This system is an AI-based screening/research tool. It is not a substitute for examination or diagnosis by a qualified eye-care professional.