import os

# Disable GPU/CUDA attempts on Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import numpy as np


app = Flask(__name__)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "eye_disease_model_FINAL.keras"

model = None


def get_model():
    global model

    if model is None:
        print("Loading eye disease model...")
        model = load_model(
            MODEL_PATH,
            compile=False
        )
        print("Model loaded successfully.")

    return model


# ============================================================
# CLASS ORDER
# IMPORTANT: Must match the training class order
# ============================================================

classes = [
    "diabetic retinopathy",
    "glaucoma",
    "healthy",
    "myopia",
    "retinis pigmentosa"
]


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None
    filename = None
    image_url = None
    error = None

    if request.method == "POST":

        if "file" not in request.files:

            error = "Please select an eye image."

        else:

            file = request.files["file"]

            if file.filename == "":

                error = "Please select an eye image."

            else:

                filename = secure_filename(file.filename)

                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                file.save(filepath)

                try:

                    # ====================================================
                    # LOAD MODEL ONLY WHEN NEEDED
                    # ====================================================

                    current_model = get_model()

                    # ====================================================
                    # IMAGE PREPROCESSING
                    # ====================================================

                    img = image.load_img(
                        filepath,
                        target_size=(224, 224)
                    )

                    img_array = image.img_to_array(img)

                    img_array = img_array / 255.0

                    img_array = np.expand_dims(
                        img_array,
                        axis=0
                    )

                    # ====================================================
                    # MODEL PREDICTION
                    # ====================================================

                    preds = current_model.predict(
                        img_array,
                        verbose=0
                    )[0]

                    # ====================================================
                    # DEBUG
                    # ====================================================

                    print()
                    print("=" * 60)
                    print("MODEL PREDICTION")
                    print("=" * 60)

                    for i, class_name in enumerate(classes):
                        print(
                            f"{class_name}: "
                            f"{float(preds[i]) * 100:.2f}%"
                        )

                    print("-" * 60)

                    predicted_index = int(
                        np.argmax(preds)
                    )

                    predicted_probability = float(
                        preds[predicted_index]
                    )

                    print(
                        "Predicted index:",
                        predicted_index
                    )

                    print(
                        "Predicted class:",
                        classes[predicted_index]
                    )

                    print(
                        "Maximum confidence:",
                        f"{predicted_probability * 100:.2f}%"
                    )

                    print("=" * 60)
                    print()

                    # ====================================================
                    # FINAL RESULT
                    # ====================================================

                    prediction = classes[predicted_index]

                    confidence = round(
                        predicted_probability * 100,
                        2
                    )

                    image_url = f"/uploads/{filename}"

                except Exception as e:

                    print("PREDICTION ERROR:", str(e))

                    error = (
                        "An error occurred while processing "
                        "the image. Please try again."
                    )


    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        image_url=image_url,
        error=error
    )


# ============================================================
# SERVE UPLOADED IMAGE
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):

    return send_from_directory(
        UPLOAD_FOLDER,
        filename
    )


# ============================================================
# RUN FLASK
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 10000)),
        debug=False
    )
