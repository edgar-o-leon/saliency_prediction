# Import required libraries
#from utils.data_process import preprocess_img, postprocess_img

import json
import numpy as np
import torch
import torchvision.transforms as transforms
import os
import logging
import base64
from PIL import Image
import io # Import the io module
import onnxruntime as ort
import cv2

def preprocess_img(image, channels=3):
    # Convert the PIL Image to a NumPy array
    img = np.array(image)

    # Ensure the image is in the correct channel format
    if channels == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif channels == 3 and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    shape_r = 288
    shape_c = 384
    img_padded = np.ones((shape_r, shape_c, channels), dtype=np.uint8)
    if channels == 1:
        img_padded = np.zeros((shape_r, shape_c), dtype=np.uint8)
    original_shape = img.shape
    rows_rate = original_shape[0] / shape_r
    cols_rate = original_shape[1] / shape_c
    if rows_rate > cols_rate:
        new_cols = (original_shape[1] * shape_r) // original_shape[0]
        img = cv2.resize(img, (new_cols, shape_r))
        if new_cols > shape_c:
            new_cols = shape_c
        img_padded[:,
        ((img_padded.shape[1] - new_cols) // 2):((img_padded.shape[1] - new_cols) // 2 + new_cols)] = img
    else:
        new_rows = (original_shape[0] * shape_c) // original_shape[1]
        img = cv2.resize(img, (shape_c, new_rows))

        if new_rows > shape_r:
            new_rows = shape_r
        img_padded[((img_padded.shape[0] - new_rows) // 2):((img_padded.shape[0] - new_rows) // 2 + new_rows),
        :] = img

    return img_padded

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def model_fn(model_dir):
    model_path = os.path.join(model_dir, 'Transalnet_best_model.onnx')
    model = ort.InferenceSession(model_path)
    logging.info(f"Model loaded from {model_path}")
    logging.info(f"Numpy version: {np.__version__}")
    logging.info(f"ONNX Runtime version: {ort.__version__}")
    return model


def input_fn(request_body, request_content_type):
    logging.debug(f"Request content type: {request_content_type}")
    logging.debug(f"Request body: {request_body}")
    
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if 'image' in data:
            encoded_image = data['image']
            image_data = base64.b64decode(encoded_image)
            image = Image.open(io.BytesIO(image_data)).convert('RGB')            
            img = preprocess_img(image)  # Padding and resizing input image into 384x288
            img = np.array(img)/255
            img = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
            img = img.astype(np.float32)
            logging.debug(f"Image decoded, size: {image.size}")
            return img
        else:
            raise ValueError("Missing 'image' key in JSON payload")
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")



def predict_fn(input_data, model):
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: input_data})
    return outputs[0]


def output_fn(prediction, response_content_type):
    if response_content_type == 'application/json':
        # Squeeze and convert to PIL Image
        pred_saliency_np = prediction.squeeze()
        toPIL = transforms.ToPILImage()
        pic = toPIL(pred_saliency_np)
        
        # Convert the PIL image back to base64-encoded string
        buffered = io.BytesIO()
        pic.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        response_body = json.dumps({'image': encoded_image})
        return response_body
    else:
        raise ValueError("Unsupported content type: {}".format(response_content_type))

