import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

def expand_borders(img, border_size=50):
    border_color = img[0, 0].tolist()
    expanded_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=border_color)
    return expanded_img


def sort_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    s = corners.sum(axis=1)
    rect[0] = corners[np.argmin(s)]  # Top-left
    rect[2] = corners[np.argmax(s)]  # Bottom-right

    diff = np.diff(corners, axis=1)
    rect[1] = corners[np.argmin(diff)]  # Top-right
    rect[3] = corners[np.argmax(diff)]  # Bottom-left

    return rect


def warp_perspective(image, corners):
    rect = sort_corners(corners)
    (tl, tr, br, bl) = rect

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    m = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, m, (max_width, max_height))

    return warped


def extend_line(x1, y1, x2, y2, img_shape):
    img_height, img_width = img_shape
    if x1 == x2:  # Vertical line
        return (x1, 0), (x1, img_height - 1)
    if y1 == y2:  # Horizontal line
        return (0, y1), (img_width - 1, y1)

    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    x_top = (0 - intercept) / slope
    x_bottom = (img_height - 1 - intercept) / slope
    y_left = slope * 0 + intercept
    y_right = slope * (img_width - 1) + intercept

    points = []
    if 0 <= x_top < img_width:
        points.append((int(x_top), 0))
    if 0 <= x_bottom < img_width:
        points.append((int(x_bottom), img_height - 1))
    if 0 <= y_left < img_height:
        points.append((0, int(y_left)))
    if 0 <= y_right < img_height:
        points.append((img_width - 1, int(y_right)))

    if len(points) == 2:
        return points[0], points[1]
    else:
        return None, None


def straighten(img, colored):
    img_expanded = expand_borders(img, border_size=20)
    img_lines = expand_borders(colored, border_size=20)
    img_lines = cv2.resize(img_lines, (500, 500))

    img_resized = cv2.resize(img_expanded, (500, 500))

    _, img_thresh = cv2.threshold(img_resized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(img_thresh, 50, 150, apertureSize=3)

    #cv2.imshow('edges', edges)
    plt.subplot(3,3,1)
    plt.imshow(edges,cmap='gray')
    plt.title('edges')

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        print("No lines detected.")
        return None

    extended_lines = []

    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)

        (x1_ext, y1_ext), (x2_ext, y2_ext) = extend_line(x1, y1, x2, y2, img_resized.shape[:2])
        if (x1_ext, y1_ext) is not None and (x2_ext, y2_ext) is not None:
            extended_lines.append(((x1_ext, y1_ext), (x2_ext, y2_ext)))
            cv2.line(img_lines, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 0, 255), 2)
           # cv2.imshow('lines', img_lines)
            plt.subplot(3,3,2)
            plt.imshow(img_lines,cmap='gray')
            plt.title('lines')

    if len(extended_lines) < 4:
        print("Not enough extended lines.")
        return None

    points = []
    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            (x1, y1), (x2, y2) = extended_lines[i]
            (x3, y3), (x4, y4) = extended_lines[j]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                if 0 <= intersect_x < img_resized.shape[1] and 0 <= intersect_y < img_resized.shape[0]:
                    points.append((int(intersect_x), int(intersect_y)))

    if len(points) < 4:
        print("Not enough intersection points.")
        return None

    points = np.array(points)
    corners = sort_corners(points)

    if corners is not None:
        for i in range(4):
            cv2.line(img_lines, tuple(map(int, corners[i])), tuple(map(int, corners[(i + 1) % 4])), (0, 255, 0),
                     2)
        warped_image = warp_perspective(img_resized, corners)
        h, w = warped_image.shape[:2]

        if w / h < 0.7:
            # Rotate 90 degrees clockwise if width is less than height
            warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)

       # cv2.imshow('Detected Card Frame', img_lines)
        #cv2.imshow('Warped Image', warped_image)
        plt.subplot(3,3,3)
        plt.imshow(img_lines,cmap='gray')
        plt.title('Detected Card Frame')
        plt.subplot(3, 3, 4)
        plt.imshow(warped_image, cmap='gray')
        plt.title('Warped Image')

        return warped_image
    else:
        return None


def crop_pan(warped_img):
    h, w = warped_img.shape[:2]

    pan_y_start_ratio = 0.52
    pan_y_end_ratio = 0.7
    pan_x_start_ratio = 0.1
    pan_x_end_ratio = 0.93

    pan_x_start = int(pan_x_start_ratio * w)
    pan_x_end = int(pan_x_end_ratio * w)
    pan_y_start = int(pan_y_start_ratio * h)
    pan_y_end = int(pan_y_end_ratio * h)

    pan_region = warped_img[pan_y_start:pan_y_end, pan_x_start:pan_x_end]

    return pan_region


def pre_process(img):
    img_colored = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_straightened = straighten(img_gray, img_colored)

    if img_straightened is None:
        print("No card detected.")
        return None

    otsu_threshold_value, img_thresh = cv2.threshold(img_straightened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Otsu's threshold value:", otsu_threshold_value)

    #cv2.imshow('threshold', img_thresh)
    plt.subplot(3, 3, 5)
    plt.imshow(img_thresh, cmap='gray')
    plt.title('Threshold')

    img_resized = cv2.resize(img_thresh, (1000, 700))

    #cv2.imshow('img_resized', img_resized)
    plt.subplot(3, 3, 6)
    plt.imshow(img_resized, cmap='gray')
    plt.title('img_resized')

    pan = crop_pan(img_resized)

    return pan


# Example usage:
input_img = cv2.imread("D:\senior 1\summer 24\computer vision\CVProject\CSE483 SMR24 Project Test Cases/06 - Hatetlewe7 hatlewe7.jpg")
pan_image = pre_process(input_img)
if pan_image is not None:
    #cv2.imshow('PAN Image', pan_image)
    plt.subplot(3, 3, 7)
    plt.imshow(pan_image, cmap='gray')
    plt.title('Pan Image')
    cv2.imwrite('pan.jpg', pan_image)
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
