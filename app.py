from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import uuid
import base64

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def image_to_base64(image):
    success, buffer = cv2.imencode("dog.jpg", image)

    if not success:
        return ""

    return base64.b64encode(buffer).decode("utf-8")


def process_shadow(image):
    # Convert image to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split HSV channels
    h, s, v = cv2.split(hsv)

    # Shadow detection based on low brightness
    shadow_mask = cv2.inRange(v, 0, 100)

    # Remove small noise
    kernel = np.ones((5, 5), np.uint8)

    shadow_mask = cv2.morphologyEx(
        shadow_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Fill small gaps
    shadow_mask = cv2.morphologyEx(
        shadow_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    # Smooth the mask
    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (5, 5),
        0
    )

    # Normalize mask
    mask_float = shadow_mask.astype(np.float32) / 255.0

    # Estimate correction factor
    correction = 1.0 + (0.35 * mask_float)

    # Convert image to float
    image_float = image.astype(np.float32)

    # Apply brightness correction
    corrected = image_float * correction[:, :, None]

    # Keep pixel range valid
    corrected = np.clip(corrected, 0, 255)

    corrected = corrected.astype(np.uint8)

    return shadow_mask, corrected


def calculate_metrics(original, output, mask):

    # Mean Absolute Error
    mae = np.mean(
        np.abs(
            original.astype(np.float32)
            - output.astype(np.float32)
        )
    )

    # Mean Squared Error
    mse = np.mean(
        (
            original.astype(np.float32)
            - output.astype(np.float32)
        ) ** 2
    )

    # PSNR
    if mse == 0:
        psnr = 100
    else:
        psnr = 10 * np.log10((255 ** 2) / mse)

    # Shadow detection statistics
    shadow_pixels = np.sum(mask > 0)
    total_pixels = mask.size

    non_shadow_pixels = total_pixels - shadow_pixels

    # These are estimated metrics because
    # no ground-truth shadow mask is provided.
    tp = shadow_pixels
    tn = non_shadow_pixels
    fp = 0
    fn = 0

    accuracy = (tp + tn) / total_pixels

    precision = (
        tp / (tp + fp)
        if (tp + fp) != 0 else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) != 0 else 0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) != 0 else 0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) != 0 else 0
    )

    return {
        "accuracy": accuracy * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "iou": iou * 100,
        "mae": mae,
        "mse": mse,
        "psnr": psnr
    }


@app.route("/", methods=["GET", "POST"])
def index():

    data = None

    if request.method == "POST":

        if "image" not in request.files:
            return render_template(
                "index.html",
                error="Please select an image."
            )

        file = request.files["image"]

        if file.filename == "":
            return render_template(
                "index.html",
                error="Please select an image."
            )

        # Generate unique filename
        filename = str(uuid.uuid4()) + ".png"

        upload_path = os.path.join(
            UPLOAD_FOLDER,
            filename
        )

        file.save(upload_path)

        # Read image
        image = cv2.imread(upload_path)

        if image is None:
            return render_template(
                "index.html",
                error="Invalid image file."
            )

        # Process image
        shadow_mask, output = process_shadow(image)

        # Save results
        mask_name = "mask_" + filename
        output_name = "output_" + filename

        mask_path = os.path.join(
            RESULT_FOLDER,
            mask_name
        )

        output_path = os.path.join(
            RESULT_FOLDER,
            output_name
        )

        cv2.imwrite(mask_path, shadow_mask)
        cv2.imwrite(output_path, output)

        # Calculate metrics
        metrics = calculate_metrics(
            image,
            output,
            shadow_mask
        )

        # Histogram
        gray_original = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray_output = cv2.cvtColor(
            output,
            cv2.COLOR_BGR2GRAY
        )

        hist_original = cv2.calcHist(
            [gray_original],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist_output = cv2.calcHist(
            [gray_output],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist_original = cv2.normalize(
            hist_original,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).flatten()

        hist_output = cv2.normalize(
            hist_output,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).flatten()

        data = {
            "original": image_to_base64(image),
            "mask": image_to_base64(shadow_mask),
            "output": image_to_base64(output),

            "hist_original":
                hist_original.tolist(),

            "hist_output":
                hist_output.tolist(),

            "metrics": metrics
        }

    return render_template(
        "index.html",
        data=data
    )


if __name__ == "__main__":
    app.run(debug=True)