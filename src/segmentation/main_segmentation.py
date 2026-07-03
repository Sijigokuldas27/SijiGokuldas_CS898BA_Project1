import cv2
import numpy as np
import os


# Create Output Folder

output_folder = os.path.join("output", "segmentation_output")
os.makedirs(output_folder, exist_ok=True)


# Load Original Image

image_path = os.path.join("data", "HW1_IMG_CS898BA.png")
image = cv2.imread(image_path)

if image is None:
    print("Error: Unable to load image.")
    exit()

print("Image loaded successfully.")
print("Image Size:", image.shape)


# Multi-Channel Histogram Equalization

blue, green, red = cv2.split(image)

blue_eq = cv2.equalizeHist(blue)
green_eq = cv2.equalizeHist(green)
red_eq = cv2.equalizeHist(red)

normalized_image = cv2.merge((blue_eq, green_eq, red_eq))

normalized_path = os.path.join(
    output_folder,
    "normalized_color_image.png"
)

cv2.imwrite(normalized_path, normalized_image)

print("Normalized image saved.")


# Convert to Grayscale

gray = cv2.cvtColor(
    normalized_image,
    cv2.COLOR_BGR2GRAY
)


# Otsu Thresholding

_, otsu_mask = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    os.path.join(output_folder, "otsu_binary_mask.png"),
    otsu_mask
)

# Foreground Extraction using Otsu
otsu_foreground = cv2.bitwise_and(
    normalized_image,
    normalized_image,
    mask=otsu_mask
)

cv2.imwrite(
    os.path.join(output_folder, "otsu_foreground.png"),
    otsu_foreground
)

print("Otsu segmentation completed.")


# Adaptive Gaussian Thresholding

adaptive_mask = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

cv2.imwrite(
    os.path.join(output_folder, "adaptive_binary_mask.png"),
    adaptive_mask
)

adaptive_foreground = cv2.bitwise_and(
    normalized_image,
    normalized_image,
    mask=adaptive_mask
)

cv2.imwrite(
    os.path.join(output_folder, "adaptive_foreground.png"),
    adaptive_foreground
)

print("Adaptive thresholding completed.")

# Convert normalized image to HSV
hsv_image = cv2.cvtColor(
    normalized_image,
    cv2.COLOR_BGR2HSV
)

print("HSV image created successfully.")

# Prepare image for K-Means
pixel_values = hsv_image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

print("Pixel values prepared for K-Means.")

# Set K-Means criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

print("K-Means criteria created.")

# K-Means with K = 3
k = 3

_, labels, centers = cv2.kmeans(
    pixel_values,
    k,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(hsv_image.shape)

segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)

cv2.imwrite(
    os.path.join(output_folder, "kmeans_k3.png"),
    segmented_bgr
)

print("K-Means (K=3) completed.")

# Set K-Means criteria
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

print("K-Means criteria created.")

# K-Means with K = 4
k = 4

_, labels, centers = cv2.kmeans(
    pixel_values,
    k,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(hsv_image.shape)

segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)

cv2.imwrite(
    os.path.join(output_folder, "kmeans_k4.png"),
    segmented_bgr
)

print("K-Means (K=4) completed.")

# K-Means with K = 5
k = 5

_, labels, centers = cv2.kmeans(
    pixel_values,
    k,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)

segmented_image = centers[labels.flatten()]
segmented_image = segmented_image.reshape(hsv_image.shape)

segmented_bgr = cv2.cvtColor(segmented_image, cv2.COLOR_HSV2BGR)

cv2.imwrite(
    os.path.join(output_folder, "kmeans_k5.png"),
    segmented_bgr
)

print("K-Means (K=5) completed.")