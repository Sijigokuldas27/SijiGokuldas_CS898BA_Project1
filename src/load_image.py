import cv2

img = cv2.imread("data/HW1_IMG_CS898BA.png")

if img is None:
    print("Image was not loaded")
else:
    print("Image loaded successfully")

    height = img.shape[0]
    width = img.shape[1]
    channels = img.shape[2]

    print("Height =", height)
    print("Width =", width)
    print("Channels =", channels)