from flask import Flask, request, jsonify
import cv2
import torch
import numpy as np
import insightface
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import os
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from insightface.model_zoo.inswapper import INSwapper
from datetime import datetime

app = Flask(__name__)

# Load the face swap model
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
# Load the face swap model


def read_image(file):
    """Convert uploaded image file to OpenCV format."""
    image = np.frombuffer(file.read(), np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


# def upscale_image(image, scale_factor=2):
#     """Upscale the image using bicubic interpolation. scale_factor 2=4k 1.5=QHD 1=HD"""
#     new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
#     upscaled_image = cv2.resize(image, new_size, interpolation=cv2.INTER_CUBIC)
#     return upscaled_image

# def upscale_to_4k(image):
#     """Upscale an image to 4K resolution (3840x2160) using bicubic interpolation."""
#     target_width = 3840
#     target_height = 2160
#     upscaled_image = cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
#     return upscaled_image

def upscale_to_4k_with_ai(img):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,  
        model_path='RealESRGAN_x4plus.pth',
        model=model,
        tile=0,
        device=device
    )
    upscaled_img, _ = upsampler.enhance(img)
    return upscaled_img


@app.route("/swap", methods=["POST"])
def swap_faces():
    if "source" not in request.files or "target" not in request.files:
        return jsonify({"error": "Missing source or target image"}), 400

    source_img = read_image(request.files["source"])
    target_img = read_image(request.files["target"])

    # Detect and swap faces
    source_faces = face_app.get(source_img)
    target_faces = face_app.get(target_img)
    print("How many faces in source_faces:", len(source_faces))
    print("How many faces in target_faces:", len(target_faces))
    if not source_faces or not target_faces:
        return jsonify({"error": "Face not detected in one of the images"}), 400
    swapper = insightface.model_zoo.get_model(
        "inswapper_128.onnx", download=True, download_zip=True
    )
    swapped_img = swapper.get(
        target_img, target_faces[0], source_faces[0], paste_back=True
    )
    upscaled_img = upscale_to_4k_with_ai(swapped_img)
    # Convert swapped image to JPEG and return
    _, buffer = cv2.imencode(".jpg", upscaled_img)
    image_bytes = buffer.tobytes()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"swapped_image_{timestamp}.jpg"
    with open(file_path, "wb") as f:
        f.write(image_bytes)

    return buffer.tobytes(), 200, {"Content-Type": "image/jpeg"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
