import cv2
import numpy as np
from scipy import stats

img = cv2.imread("data/HW1_IMG_CS898BA.png")

if img is None:
    print("Image was not loaded")

else:
    print("Image loaded")

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print("Height =", height)
    print("Width =", width)
    print("Channels =", channels)

    channel_names = ["Blue", "Green", "Red"]

    for i in range(3):
        pixels = img[:, :, i].flatten()

        print("\n" + channel_names[i] + " Channel")
        print("Min:", np.min(pixels))
        print("Max:", np.max(pixels))
        print("Average:", np.mean(pixels))
        print("Median:", np.median(pixels))
        print("Mode:", stats.mode(pixels, keepdims=True).mode[0])
        print("Skew:", stats.skew(pixels))
        print("Range:", np.ptp(pixels))
        print("Standard Deviation:", np.std(pixels))
        print("Variance:", np.var(pixels))

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output/grayscale/grayscale.png", gray)

    # Binary
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("output/binary/binary.png", binary)

    # HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite("output/hsv/hsv.png", hsv)

    # LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cv2.imwrite("output/lab/lab.png", lab)

    # HLS
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    cv2.imwrite("output/hls/hls.png", hls)

    # Normalized HSV
    hsv_norm = hsv.copy()
    hsv_norm[:, :, 2] = cv2.equalizeHist(hsv_norm[:, :, 2])

    cv2.imwrite("output/hsv/hsv_normalized.png", hsv_norm)

    rgb_normalized = cv2.cvtColor(hsv_norm, cv2.COLOR_HSV2BGR)
    cv2.imwrite("output/hsv/rgb_normalized.png", rgb_normalized)

    print("\nConverted images saved.")
    # Affine Transformation - Translation

rows, cols = gray.shape

translation_matrix = np.float32([
    [1, 0, 50],
    [0, 1, 30]
])

translated = cv2.warpAffine(
    gray,
    translation_matrix,
    (cols, rows)
)

cv2.imwrite(
    "output/affine/grayscale_translation.png",
    translated
)

print("First affine image saved.")
rotation_matrix = cv2.getRotationMatrix2D(
    (cols / 2, rows / 2),
    45,
    1
)

rotated = cv2.warpAffine(
    gray,
    rotation_matrix,
    (cols, rows)
)

cv2.imwrite(
    "output/affine/grayscale_rotation.png",
    rotated
)

print("Second affine image saved.")
rotation_matrix = cv2.getRotationMatrix2D(
    (cols / 2, rows / 2),
    45,
    1
)

rotated = cv2.warpAffine(
    gray,
    rotation_matrix,
    (cols, rows)
)

cv2.imwrite(
    "output/affine/grayscale_rotation.png",
    rotated
)

print("Second affine image saved.")
# Binary Transformations
translated = cv2.warpAffine(binary, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/binary_translation.png", translated)

rotated = cv2.warpAffine(binary, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/binary_rotation.png", rotated)

# HSV Transformations
translated = cv2.warpAffine(hsv, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/hsv_translation.png", translated)

rotated = cv2.warpAffine(hsv, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/hsv_rotation.png", rotated)

# HSV Normalized Transformations
translated = cv2.warpAffine(hsv_norm, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/hsv_normalized_translation.png", translated)

rotated = cv2.warpAffine(hsv_norm, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/hsv_normalized_rotation.png", rotated)

# RGB Normalized Transformations
translated = cv2.warpAffine(rgb_normalized, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/rgb_normalized_translation.png", translated)

rotated = cv2.warpAffine(rgb_normalized, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/rgb_normalized_rotation.png", rotated)

# LAB Transformations
translated = cv2.warpAffine(lab, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/lab_translation.png", translated)

rotated = cv2.warpAffine(lab, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/lab_rotation.png", rotated)

# HLS Transformations
translated = cv2.warpAffine(hls, translation_matrix, (cols, rows))
cv2.imwrite("output/affine/hls_translation.png", translated)

rotated = cv2.warpAffine(hls, rotation_matrix, (cols, rows))
cv2.imwrite("output/affine/hls_rotation.png", rotated)

print("All affine transformations saved.")
# Gaussian Blur on Grayscale Image

sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        gray,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/grayscale_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("Grayscale blur images saved.")

# Gaussian Blur on Binary Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        binary,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/binary_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("Binary blur images saved.")


# Gaussian Blur on HSV Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        hsv,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/hsv_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("HSV blur images saved.")

# Gaussian Blur on HSV Normalized Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        hsv_norm,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/hsv_normalized_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("HSV Normalized blur images saved.")


# Gaussian Blur on RGB Normalized Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        rgb_normalized,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/rgb_normalized_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("RGB Normalized blur images saved.")


# Gaussian Blur on LAB Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        lab,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/lab_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("LAB blur images saved.")


# Gaussian Blur on HLS Image

for sigma in sigmas:

    blurred = cv2.GaussianBlur(
        hls,
        (0, 0),
        sigma
    )

    filename = (
        "output/blur/hls_sigma_"
        + str(sigma)
        + ".png"
    )

    cv2.imwrite(
        filename,
        blurred
    )

print("HLS blur images saved.")
import os

sigmas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

affine_folder = "output/affine"

for image_name in os.listdir(affine_folder):

    image_path = os.path.join(affine_folder, image_name)

    image = cv2.imread(image_path)

    if image is None:
        continue

    base_name = image_name.replace(".png", "")

    for sigma in sigmas:

        blurred = cv2.GaussianBlur(
            image,
            (0, 0),
            sigma
        )

        save_name = (
            "output/blur/"
            + base_name
            + "_sigma_"
            + str(sigma)
            + ".png"
        )

        cv2.imwrite(
            save_name,
            blurred
        )

print("Affine blur images saved.")

import os
import random
import shutil

all_images = []

for folder in ["affine", "blur"]:

    folder_path = "output/" + folder

    for file in os.listdir(folder_path):

        if file.endswith(".png"):

            all_images.append(
                os.path.join(folder_path, file)
            )

random.shuffle(all_images)

subset_size = 42

for i in range(4):

    subset_folder = "output/subset" + str(i + 1)

    start = i * subset_size
    end = start + subset_size

    subset_images = all_images[start:end]

    for image_file in subset_images:

        shutil.copy(
            image_file,
            subset_folder
        )

print("Subsets created.")

# Sobel Edge Detection

import os

subset_folder = "output/subset1"

for file in os.listdir(subset_folder):

    image_path = os.path.join(
        subset_folder,
        file
    )

    image = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        continue

    sobel_x = cv2.Sobel(
        image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    sobel_y = cv2.Sobel(
        image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    sobel = cv2.magnitude(
        sobel_x,
        sobel_y
    )

    save_name = os.path.join(
        "output/sobel",
        "sobel_" + file
    )

    cv2.imwrite(
        save_name,
        sobel
    )

print("Sobel images saved.")