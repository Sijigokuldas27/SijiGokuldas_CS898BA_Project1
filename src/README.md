# CS898BA Project 1

## Student Information

Name: Siji Gokuldas

## Project Overview

This project was completed for CS898BA. The goal of the assignment was to perform image analysis and image processing using Python and OpenCV. Different image transformations, color space conversions, blurring techniques, and edge detection methods were applied to the provided image.

## Software and Libraries Used

* Python
* OpenCV
* NumPy
* Matplotlib

## How to Run the Program

1. Install the required libraries listed in requirements.txt.
2. Open the project folder in VS Code.
3. Open a terminal in the project directory.
4. Run the following command:

python src/load_image.py

## Tasks Completed

* Calculated image statistics
* Converted the image to grayscale
* Created a binary image
* Converted the image to HSV, LAB, and HLS color spaces
* Applied histogram equalization on the HSV image
* Converted the normalized image back to RGB
* Performed affine transformations
* Applied Gaussian blur with different sigma values
* Created four random subsets
* Applied Sobel edge detection
* Applied Laplacian edge detection
* Applied Canny edge detection
* Applied Prewitt edge detection

## Results

The project successfully generated all required output images. Different image processing techniques were used to analyze and improve the image. The edge detection methods produced different results, allowing comparison of their performance on the dataset.

## Gaussian Blur Analysis

Gaussian blur was applied using sigma values of 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, and 3.5. As the sigma value increased, the images became smoother and less detailed. Lower sigma values preserved most image features while reducing minor noise. Higher sigma values produced stronger blurring and removed more image details.

## Edge Detection Analysis

Four edge detection techniques were applied to the selected subset of images.

### Sobel
Sobel detected major edges and object boundaries clearly while maintaining a good balance between detail and noise reduction.

### Laplacian
Laplacian detected fine intensity changes but was more sensitive to noise and produced additional edge responses in some images.

### Canny
Canny generated thin and precise edges. However, due to the dark nature of the original image, some important details were not consistently detected.

### Prewitt
Prewitt produced results similar to Sobel but with slightly weaker edge responses and fewer detected details.

## Best Performing Method

Based on visual comparison of the generated outputs, Sobel provided the most useful and consistent edge detection results for this dataset. It highlighted the major outlines of objects while reducing unnecessary noise.

## Conclusion

This project demonstrated image analysis and image processing techniques using Python and OpenCV. The assignment included image statistics, color space conversion, histogram equalization, affine transformations, Gaussian blurring, image subset creation, and edge detection. The generated outputs showed how different techniques affect image quality and feature extraction. Overall, Sobel produced the most effective edge detection results for the selected image subset.
