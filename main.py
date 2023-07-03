import cv2
import numpy as np
from sklearn.cluster import KMeans

def color_detection(image_path, num_colors):
    # Load the image
    image = cv2.imread(image_path)
    
    # Reshape the image to 2D array of pixels
    pixels = image.reshape(-1, 3)
    
    # Perform K-means clustering to find the dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(pixels)
    
    # Get the RGB values of the cluster centers (dominant colors)
    colors = kmeans.cluster_centers_
    
    return colors.astype(int)

# Test the color detection function
image_path = 'path_to_your_image'
num_colors = 5  # Number of dominant colors to detect

dominant_colors = color_detection(image_path, num_colors)
print("Dominant Colors:")
for color in dominant_colors:
    print(color)
