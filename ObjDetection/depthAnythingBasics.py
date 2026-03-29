from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import httpx
from io import BytesIO
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

image = Image.open('car1.png').convert("RGB")

plt.imshow(image)
plt.axis('off') 
plt.show()

image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-small-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-small-hf")

# prepare image for the model
inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

# interpolate to original size
post_processed_output = image_processor.post_process_depth_estimation(
    outputs,
    target_sizes=[(image.height, image.width)],
)

# visualize the prediction
predicted_depth = post_processed_output[0]["predicted_depth"]
depth = predicted_depth * 255 / predicted_depth.max()
depth = depth.detach().cpu().numpy()
depth = Image.fromarray(depth.astype("uint8"))

plt.imshow(depth)
plt.axis('off') 
plt.show()