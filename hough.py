import cv2
import numpy as np


def expand_borders(image, border_size=50):
    # Expand the image borders to avoid edge detection issues
    expanded_image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])  # Use a white border
    return expanded_image


def preprocess_image(blurred):
    # Apply Canny Edge Detection
    edges = cv2.Canny(blurred, 30, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area and keep the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    if len(contours) == 0:
        return None, None

    # Apply a convex hull to the largest contour to ensure it's properly enclosed
    hull = cv2.convexHull(contours[0])

    # Approximate the hull to a polygon
    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    # Ensure the polygon has four corners
    if len(approx) != 4:
        return None, None

    return approx.reshape(4, 2), hull


def sort_corners(corners):
    # Sort the corners to always be in the order: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left has the smallest sum
    rect[2] = corners[np.argmax(s)]  # Bottom-right has the largest sum

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right has the smallest difference
    rect[3] = corners[np.argmax(diff)]  # Bottom-left has the largest difference

    return rect


def warp_perspective(image, corners):
    # Ensure corners are sorted
    rect = sort_corners(corners)

    # Compute the width and height of the card
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # Destination points
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Get the perspective transform matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # Apply the perspective warp
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


# Load the image
image_path = 'CSE483 SMR24 Project Test Cases/07 - Hatet3eweg hat3eweg.jpg'
image = cv2.imread(image_path)
expanded_image = expand_borders(image)

# Convert to LAB color space
lab = cv2.cvtColor(expanded_image, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(lab)

# Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
cl = clahe.apply(l)

# Merge channels and convert back to BGR
limg = cv2.merge((cl, a, b))
enhanced_image = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)

cv2.imshow('enh',enhanced_image)

# Convert to grayscale
gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

corners, hull = preprocess_image(blurred)

if corners is not None:
    # Warp the perspective to obtain a top-down view of the card
    warped_image = warp_perspective(expanded_image, corners)

    # Display the results
    cv2.drawContours(expanded_image, [hull], -1, (0, 255, 0), 2)
    cv2.imshow('Detected Card', expanded_image)
    cv2.imshow('Warped Image', warped_image)

    gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur first
    gaussian_blurred = cv2.GaussianBlur(gray_warped, (5, 5), 0)

    # Apply Median Blur second
    median_blurred = cv2.medianBlur(gaussian_blurred, 5)

    binary = cv2.adaptiveThreshold(median_blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Apply morphological operations carefully
    img_eroded = cv2.erode(binary, kernel)
    img_dilated = cv2.dilate(img_eroded, kernel)

    # Show the preprocessed image
    cv2.imshow('Preprocessed Image', img_dilated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Card could not be detected correctly.")
