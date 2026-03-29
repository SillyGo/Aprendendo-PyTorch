from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import numpy as np
import cv2

warnings.filterwarnings('ignore')

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

def RunDepthEstimation(image):
    inputs = image_processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    post_processed_output = image_processor.post_process_depth_estimation(
        outputs,
        target_sizes=[(image.height, image.width)],
    )

    predicted_depth = post_processed_output[0]["predicted_depth"]
    depth = predicted_depth * 255 / predicted_depth.max()
    depth = depth.detach().cpu().numpy()

    return Image.fromarray(depth.astype("uint8"))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    depth = np.array(RunDepthEstimation(Image.fromarray(frame)))

    cv2.imshow('Profundidade monocular da webcam', depth)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()