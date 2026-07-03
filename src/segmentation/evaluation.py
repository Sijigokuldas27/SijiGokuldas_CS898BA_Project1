import cv2
import numpy as np
import os

# File paths
ground_truth_path = os.path.join(
    "output",
    "ground_truth",
    "ground_truth_mask.png"
)

otsu_path = os.path.join(
    "output",
    "segmentation_output",
    "otsu_binary_mask.png"
)

adaptive_path = os.path.join(
    "output",
    "segmentation_output",
    "adaptive_binary_mask.png"
)

kmeans_path = os.path.join(
    "output",
    "segmentation_output",
    "kmeans_k5.png"
)
# Load images
ground_truth = cv2.imread(
    ground_truth_path,
    cv2.IMREAD_GRAYSCALE
)

otsu = cv2.imread(
    otsu_path,
    cv2.IMREAD_GRAYSCALE
)

adaptive = cv2.imread(
    adaptive_path,
    cv2.IMREAD_GRAYSCALE
)

kmeans = cv2.imread(
    kmeans_path,
    cv2.IMREAD_GRAYSCALE
)

print("Images loaded successfully.")
# Convert images to binary

_, ground_truth = cv2.threshold(
    ground_truth,
    127,
    255,
    cv2.THRESH_BINARY
)

_, otsu = cv2.threshold(
    otsu,
    127,
    255,
    cv2.THRESH_BINARY
)

_, adaptive = cv2.threshold(
    adaptive,
    127,
    255,
    cv2.THRESH_BINARY
)

_, kmeans = cv2.threshold(
    kmeans,
    127,
    255,
    cv2.THRESH_BINARY
)

print("Binary images created.")
# Function to calculate IoU
def calculate_iou(mask1, mask2):

    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    iou = np.sum(intersection) / np.sum(union)

    return iou

print("IoU function created.")
# Calculate IoU values

otsu_iou = calculate_iou(
    ground_truth,
    otsu
)

adaptive_iou = calculate_iou(
    ground_truth,
    adaptive
)

kmeans_iou = calculate_iou(
    ground_truth,
    kmeans
)

print("Otsu IoU:", otsu_iou)
print("Adaptive IoU:", adaptive_iou)
print("K-Means IoU:", kmeans_iou)
# Function to calculate Dice Coefficient
def calculate_dice(mask1, mask2):

    intersection = np.logical_and(mask1, mask2)

    dice = (2 * np.sum(intersection)) / (
        np.sum(mask1) + np.sum(mask2)
    )

    return dice

print("Dice function created.")
# Calculate Dice values

otsu_dice = calculate_dice(
    ground_truth,
    otsu
)

adaptive_dice = calculate_dice(
    ground_truth,
    adaptive
)

kmeans_dice = calculate_dice(
    ground_truth,
    kmeans
)

# Print all the values

print("\nFinal Results")

print("Otsu")
print("IoU =", otsu_iou)
print("Dice =", otsu_dice)

print()

print("Adaptive")
print("IoU =", adaptive_iou)
print("Dice =", adaptive_dice)

print()

print("K-Means")
print("IoU =", kmeans_iou)
print("Dice =", kmeans_dice)