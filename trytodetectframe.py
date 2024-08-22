import numpy as np
import cv2


def expand_borders(img, border_size=20):
    # Create a new image with expanded borders
    border_color = img[0, 0].tolist()
    expanded_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=border_color)

    return expanded_img


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


def adjust_brightness_contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b = cv2.split(lab)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L-channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge the channels back
    limg = cv2.merge((cl, a, b))

    # Convert back to BGR
    final_img = cv2.cvtColor(limg, cv2.COLOR_Lab2BGR)
    return final_img

def color_correct(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the mean and standard deviation of the grayscale image
    mean_brightness = np.mean(img_gray)
    stddev_brightness = np.std(img_gray)

    # Determine if the image is too dark or too bright
    if mean_brightness < 50 or mean_brightness > 200 or stddev_brightness < 50:
        # Apply histogram equalization and brightness adjustment only if necessary
        img_corrected = adjust_brightness_contrast(img)
        img_gray = cv2.cvtColor(img_corrected, cv2.COLOR_BGR2GRAY)

    # Thresholding to remove noise and focus on prominent features
    _, img_thresh = cv2.threshold(img_gray, np.mean(img_gray), 255, cv2.THRESH_BINARY)

    # Invert colors if the image has a dark background and white text
    if np.mean(img_thresh) < 127:
        img_gray = 255 - img_gray

    return img_gray



def remove_periodic_noise(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    fshift_abs = np.abs(fshift)
    fshift_med20 = 20 * cv2.medianBlur(fshift_abs.astype('float32'), 5)

    mask = fshift_abs > fshift_med20
    fshift[mask] = fshift_med20[mask] / 20

    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.real(img_back)
    img_back_normal = cv2.normalize(img_back, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
    return img_back_normal


def straighten(img):
    # Expand the borders to help with detection
    img_expanded = expand_borders(img, border_size=50)

    # Threshold to find the bounding box more effectively
    _, img_thresh = cv2.threshold(img_expanded, np.mean(img_expanded), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_reverse = 255 - img_thresh

    cv2.imshow('thresh Image', img_thresh)
    cv2.imshow('reverse Image', img_reverse)

    contours, _ = cv2.findContours(img_reverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return img

    # Draw all contours for debugging
    debug_img = img_expanded.copy()
    cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)

    largest_contour = max(contours, key=cv2.contourArea)

    # Draw the largest contour for debugging
    cv2.drawContours(debug_img, [largest_contour], -1, (255, 0, 0), 2)
    rect = cv2.minAreaRect(largest_contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Draw the bounding box for debugging
    cv2.drawContours(debug_img, [box], 0, (0, 0, 255), 2)

    # Display the debug image
    cv2.imshow('Contours and Bounding Box', debug_img)

    hull = cv2.convexHull(largest_contour)

    epsilon = 0.02 * cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, epsilon, True)

    if len(approx) != 4:
        return None, None

    corners = approx.reshape(4, 2)

    if corners is not None:
        # Warp the perspective to obtain a top-down view of the card
        warped_image = warp_perspective(img_expanded, corners)

        # Display the results
        cv2.drawContours(img_expanded, [hull], -1, (0, 255, 0), 2)
        cv2.imshow('Detected Card', img_expanded)
        cv2.imshow('Warped Image', warped_image)

    return warped_image


def pre_process(img):
    img_gray_before = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray before', img_gray_before)
    img_gray = color_correct(img)
    cv2.imshow('gray Image', img_gray)
    img_noise_removed = remove_periodic_noise(img_gray)
    cv2.imshow('noise Image', img_noise_removed)
    # img_resized = cv2.resize(img_noise_removed, (1000, 700))
    # cv2.imshow('resized Image', img_resized)
    img_straightened = straighten(img_noise_removed)
    cv2.imshow('straight Image', img_straightened)

    img_median = cv2.medianBlur(img_straightened, 3)
    if np.mean(abs(img_straightened - img_median)) > 85:
        img_straightened = img_median

    cv2.imshow('median Image', img_straightened)

    img_blur = cv2.GaussianBlur(img_straightened, (5, 5), 0)

    cv2.imshow('gaus Image', img_blur)

    img_thresh = cv2.adaptiveThreshold(img_blur, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 7)

    cv2.imshow('threshold Image', img_thresh)

    square_se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilated = ~cv2.dilate(~img_thresh, square_se)
    img_closed = ~cv2.erode(~img_dilated, square_se)
    img_eroded = ~cv2.erode(~img_closed, square_se)
    img_opened = ~cv2.dilate(~img_eroded, square_se)

    return img_opened


# Example usage with an input image:
input_image = cv2.imread(
    'CSE483 SMR24 Project Test Cases/07 - Hatet3eweg hat3eweg.jpg')
processed_image = pre_process(input_image)

# To visualize the processed image
cv2.imshow('Processed Image', processed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

######################## GOOD RESULT DAY12#######################
