import os

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
# LOAD FINAL EVALUATED MODEL
# ============================================================

model = load_model("eye_disease_model_FINAL.keras")


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

    # --------------------------------------------------------
    # When user uploads an image
    # --------------------------------------------------------

    if request.method == "POST":

        # Check whether file exists
        if "file" not in request.files:

            error = "Please select an eye image."

        else:

            file = request.files["file"]

            # Check whether filename is empty
            if file.filename == "":

                error = "Please select an eye image."

            else:

                # Make filename safe
                filename = secure_filename(file.filename)

                # File path
                filepath = os.path.join(
                    UPLOAD_FOLDER,
                    filename
                )

                # Save uploaded image
                file.save(filepath)

                # ------------------------------------------------
                # IMAGE PREPROCESSING
                # ------------------------------------------------

                img = image.load_img(
                    filepath,
                    target_size=(224, 224)
                )

                img_array = image.img_to_array(img)

                # Same normalization used during training
                img_array = img_array / 255.0

                # Add batch dimension
                img_array = np.expand_dims(
                    img_array,
                    axis=0
                )

                # ------------------------------------------------
                # MODEL PREDICTION
                # ------------------------------------------------

                preds = model.predict(
                    img_array,
                    verbose=0
                )[0]

                # ------------------------------------------------
                # DEBUG / VERIFICATION
                # ------------------------------------------------

                print()
                print("=" * 60)
                print("MODEL PREDICTION")
                print("=" * 60)

                print("Raw probabilities:")

                for i, class_name in enumerate(classes):
                    print(
                        f"{class_name}: "
                        f"{float(preds[i]) * 100:.2f}%"
                    )

                print("-" * 60)

                # Find class with highest probability
                predicted_index = int(
                    np.argmax(preds)
                )

                # Actual probability of predicted class
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

                # ------------------------------------------------
                # FINAL PREDICTION
                # ------------------------------------------------

                prediction = classes[predicted_index]

                # IMPORTANT:
                # This is confidence for THIS uploaded image,
                # NOT the overall validation accuracy.

                confidence = round(
                    predicted_probability * 100,
                    2
                )

                # Image URL for webpage
                image_url = f"/uploads/{filename}"

    # --------------------------------------------------------
    # SEND DATA TO HTML
    # --------------------------------------------------------

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
        debug=True
    )