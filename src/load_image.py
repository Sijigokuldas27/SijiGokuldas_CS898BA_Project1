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