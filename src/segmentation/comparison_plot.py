import cv2
import matplotlib.pyplot as plt
import os

# File paths
original_path = os.path.join("data", "HW1_IMG_CS898BA.png")
normalized_path = os.path.join("output", "segmentation_output", "normalized_color_image.png")
otsu_path = os.path.join("output", "segmentation_output", "otsu_binary_mask.png")
adaptive_path = os.path.join("output", "segmentation_output", "adaptive_binary_mask.png")
kmeans_path = os.path.join("output", "segmentation_output", "kmeans_k5.png")
# Read the images
original = cv2.imread(original_path)
normalized = cv2.imread(normalized_path)
otsu = cv2.imread(otsu_path, cv2.IMREAD_GRAYSCALE)
adaptive = cv2.imread(adaptive_path, cv2.IMREAD_GRAYSCALE)
kmeans = cv2.imread(kmeans_path)

# Convert BGR images to RGB for display
original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
kmeans = cv2.cvtColor(kmeans, cv2.COLOR_BGR2RGB)

print("Images loaded successfully.")
# Display all images
plt.figure(figsize=(18, 6))

plt.subplot(1, 5, 1)
plt.imshow(original)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 5, 2)
plt.imshow(normalized)
plt.title("Normalized")
plt.axis("off")

plt.subplot(1, 5, 3)
plt.imshow(otsu, cmap="gray")
plt.title("Otsu")
plt.axis("off")

plt.subplot(1, 5, 4)
plt.imshow(adaptive, cmap="gray")
plt.title("Adaptive")
plt.axis("off")

plt.subplot(1, 5, 5)
plt.imshow(kmeans)
plt.title("K-Means")
plt.axis("off")
# Save the comparison figure
plt.tight_layout()

save_path = os.path.join("output", "comparison_plot.png")
plt.savefig(save_path)

print("Comparison figure saved.")

plt.show()