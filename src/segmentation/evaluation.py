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

    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)

    iou = np.sum(intersection) / np.sum(union)

    return iou

print("IoU function created.")

# Function to fix masks that may be inverted (black/white swapped)
def fix_if_inverted(mask, reference):

    iou_normal = calculate_iou(reference, mask)

    inverted_mask = cv2.bitwise_not(mask)
    iou_inverted = calculate_iou(reference, inverted_mask)

    if iou_inverted > iou_normal:
        print("Mask was inverted. Flipping black/white to correct it.")
        return inverted_mask
    else:
        return mask

print("Inversion-check function created.")

# Check and fix each mask against the ground truth
otsu = fix_if_inverted(otsu, ground_truth)
adaptive = fix_if_inverted(adaptive, ground_truth)
kmeans = fix_if_inverted(kmeans, ground_truth)

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

    mask1_bool = mask1 > 0
    mask2_bool = mask2 > 0

    intersection = np.logical_and(mask1_bool, mask2_bool)

    dice = (2 * np.sum(intersection)) / (
        np.sum(mask1_bool) + np.sum(mask2_bool)
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