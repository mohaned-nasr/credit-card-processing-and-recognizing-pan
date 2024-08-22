import numpy as np
import cv2
import matplotlib.pyplot as plt
import itertools
import math
def expand_borders(img, border_size=20):
    # Create a new image with expanded borders
    border_color = img[0, 0].tolist()
    expanded_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=border_color)
    return expanded_img

def find_intersection(line1, line2):
    # Unpack line information
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]

    # Coefficients of line 1
    a1 = np.cos(theta1)
    b1 = np.sin(theta1)
    x1 = a1 * rho1
    y1 = b1 * rho1

    # Coefficients of line 2
    a2 = np.cos(theta2)
    b2 = np.sin(theta2)
    x2 = a2 * rho2
    y2 = b2 * rho2

    # Solve linear equations to find intersection
    det = a1 * b2 - a2 * b1
    if det == 0:
        return None  # Lines are parallel
    x = (b2 * x1 - b1 * x2) / det
    y = (a1 * y2 - a2 * y1) / det
    return (int(np.round(x)), int(np.round(y)))

def sort_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left
    rect[2] = corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right
    rect[3] = corners[np.argmax(diff)]  # Bottom-left

    return rect


# Read the image
img12 = cv2.imread("D:\senior 1\summer 24\computer vision\CVProject\CSE483 SMR24 Project Test Cases/12 - weewooweewoo.jpg", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img12, 50, 150)
lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=190, lines=None, min_theta=0, max_theta=np.pi)
img_hough = cv2.cvtColor(img12, cv2.COLOR_GRAY2BGR)

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        # Convert polar coordinates to Cartesian coordinates
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 2000 * (-b))
        y1 = int(y0 + 2000 * (a))
        x2 = int(x0 - 2000 * (-b))
        y2 = int(y0 - 2000 * (a))

        # Draw the lines on the image
        cv2.line(img_hough, (x1, y1), (x2, y2), (0, 255, 0), 2)
if len(lines) >= 4:
    # Pick four lines from the result to form the four edges of the card
    line_pairs = [(lines[i], lines[j]) for i in range(len(lines)) for j in range(i + 1, len(lines))]

    corners = []
    for pair in line_pairs:
        intersection = find_intersection(pair[0], pair[1])
        if intersection is not None:
            corners.append(intersection)

    # Sort corners to form a rectangle using the provided sort_corners function
    if len(corners) >= 4:
        corners = np.array(corners, dtype="float32")
        sorted_corners = sort_corners(corners)

        # Calculate the width and height of the new image
        width = max(int(np.linalg.norm(sorted_corners[2] - sorted_corners[3])),
                    int(np.linalg.norm(sorted_corners[1] - sorted_corners[0])))
        height = max(int(np.linalg.norm(sorted_corners[1] - sorted_corners[2])),
                     int(np.linalg.norm(sorted_corners[0] - sorted_corners[3])))

        # Define the destination points for the perspective transform
        dst = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]], dtype="float32")

        # Get the perspective transform matrix
        M = cv2.getPerspectiveTransform(sorted_corners, dst)

        # Apply the perspective transformation to isolate the card
        warped = cv2.warpPerspective(img12, M, (width, height))




# Get the image dimensions
rows, cols = warped.shape

# Apply the Fourier Transform
dft = cv2.dft(np.float32(warped), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Create a mask with a notch filter
mask = np.ones((rows, cols, 2), np.uint8)
center_row, center_col = rows // 2, cols // 2
radius = 20 # Radius of the notch filter
cv2.circle(mask, (center_col, center_row), radius, (0, 0, 0), -1)

# Apply the mask to the Fourier-transformed image
fshift = dft_shift * mask

# Inverse Fourier Transform
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize the result
cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)
brightening=2
img_back=cv2.multiply(img_back,brightening)

warped=cv2.resize(img_back,(1000,700))
x, y, w, h = 25, 375, 900, 100  # Example coordinates; you'll need to adjust these

# Crop the PAN region
pan_region5 = warped[y:y+h, x:x+w]
histogram=cv2.calcHist([pan_region5],[0],None,[256],[0,256])
pan_region1=cv2.threshold(pan_region5, 60, 255, cv2.THRESH_BINARY_INV)[1]
histogram2=cv2.calcHist([pan_region1],[0],None,[256],[0,256])

pan_thresh=cv2.threshold(pan_region5, 60, 255, cv2.THRESH_BINARY)[1]


histogram=cv2.calcHist([img12],[0],None,[256],[0,256])
histogram2=cv2.calcHist([img_back],[0],None,[256],[0,256])
histogram3=cv2.calcHist([pan_region1],[0],None,[256],[0,256])


# Display the original and filtered images
plt.figure(figsize=(20, 10))
plt.subplot(3, 3, 1)
plt.imshow(img12, cmap='gray')
plt.title('Original Image')
plt.subplot(3, 3,2)
plt.imshow(edges, cmap='gray')
plt.title('Edges')
plt.subplot(3, 3, 3)
plt.imshow(img_hough, cmap='gray')
plt.title('Hough Lines')
plt.subplot(3, 3, 4)
plt.imshow(warped, cmap='gray')
plt.title('Isolated Card')
plt.subplot(3, 3, 5)
plt.imshow(img_back, cmap='gray')
plt.title('Filtered Image')
plt.subplot(3, 3, 6)
plt.imshow(pan_region5, cmap='gray')
plt.title('PAN Region')
plt.subplot(3,3,7)
plt.plot(histogram3)
plt.title('Histogram')



cv2.imwrite('pan.jpg', pan_region5)
plt.show()

