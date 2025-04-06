import cv2
import numpy as np
import requests
from PIL import Image
from io import BytesIO

def prewitt_edge_detection(image):
    horizontal_mask = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
    
    vertical_mask = np.array([[-1, -1, -1],
                               [ 0,  0,  0],
                               [ 1,  1,  1]])
    
    img_h = cv2.filter2D(image, -1, horizontal_mask)
    img_v = cv2.filter2D(image, -1, vertical_mask)

    gradient_magnitude = np.sqrt(img_h**2 + img_v**2).astype(np.uint8)
    _, edge_image = cv2.threshold(gradient_magnitude, 10, 255, cv2.THRESH_BINARY)
    
    return edge_image

def sobel(image):
    horizontal_mask = np.array([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]])
    
    vertical_mask = np.array([[-1, -2, -1],
                               [ 0,  0,  0],
                               [ 1,  2,  1]])
    
    img_h = cv2.filter2D(image, -1, horizontal_mask)
    img_v = cv2.filter2D(image, -1, vertical_mask)

    gradient_magnitude = np.sqrt(img_h**2 + img_v**2).astype(np.uint8)
    _, edge_image = cv2.threshold(gradient_magnitude, 10, 255, cv2.THRESH_BINARY)

    return edge_image

def difference_of_gaussians(image, sigma1=1.0, sigma2=2.0):
    blur1 = cv2.GaussianBlur(image, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(image, (0, 0), sigma2)
    
    dog = cv2.absdiff(blur1, blur2)
    dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX)

    _, edge_image = cv2.threshold(dog.astype(np.uint8), 10, 255, cv2.THRESH_BINARY)
    
    return edge_image

def load_image_from_url(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Could not retrieve image from the URL.")
    
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = np.array(img)
    
    # Convert RGB to BGR (OpenCV format)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
