import numpy as np
import cv2
import math


def generate_edge(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    new_distance = distance * 1.58
    slope = (y2 - y1) / (x2 - x1)
    perpend_slope = -1 / slope
    angle = math.atan(perpend_slope)
    print(distance)
    print(slope)
    x1_new = int(x1 + new_distance * math.cos(angle))
    y1_new = int(y1 + new_distance * math.sin(angle))
    x2_new = int(x2 + new_distance * math.cos(angle))
    y2_new = int(y2 + new_distance * math.sin(angle))

    return x1_new, y1_new, x2_new, y2_new


def expand_borders(img, border_size=50):
    border_color = img[0, 0].tolist()
    expanded_img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size,
                                      cv2.BORDER_CONSTANT, value=border_color)
    return expanded_img


def sort_corners(corners):
    rect = np.zeros((4, 2), dtype="float32")
    rect[0] = corners[0]  # Top-left
    rect[2] = corners[1]  # Bottom-right

    rect[1] = corners[3]  # Top-right
    rect[3] = corners[2]  # Bottom-left

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

    # Compute the perspective transform matrix
    m = cv2.getPerspectiveTransform(rect, dst)

    # Warp the image using the perspective transform
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
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    blur = cv2.medianBlur(gaussian, 3)

    _, img_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    edges = cv2.Canny(img_thresh, 30, 150, apertureSize=3)

    cv2.imshow('edges', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 120)
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
        (x1_ext, y1_ext), (x2_ext, y2_ext) = extend_line(x1, y1, x2, y2, img.shape[:2])
        if (x1_ext, y1_ext) is not None and (x2_ext, y2_ext) is not None:
            extended_lines.append(((x1_ext, y1_ext), (x2_ext, y2_ext)))
            cv2.line(colored, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 0, 255), 2)

    del (extended_lines[0])
    point1 = (243, 14)
    point2 = (87, 149)
    x1gen, y1gen, x2gen, y2gen = generate_edge(point1, point2)
    (x1_new, y1_new), (x2_new, y2_new) = extend_line(x1gen, y1gen, x2gen, y2gen, img.shape[:2])
    extended_lines.append(((x1_new, y1_new), (x2_new, y2_new)))
    cv2.line(colored, (x1_new, y1_new), (x2_new, y2_new), (0, 0, 255), 2)
    cv2.imshow('lines', colored)

    points = []
    for i in range(len(extended_lines)):
        for j in range(i + 1, len(extended_lines)):
            (x1, y1), (x2, y2) = extended_lines[i]
            (x3, y3), (x4, y4) = extended_lines[j]

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if denom != 0:
                intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                if 0 <= intersect_x < img.shape[1] and 0 <= intersect_y < img.shape[0]:
                    points.append((int(intersect_x), int(intersect_y)))
    print(points)

    if len(points) < 4:
        print("Not enough intersection points.")
        return None

    points = np.array(points)
    corners = sort_corners(points)

    if corners is not None:
        for i in range(4):
            cv2.line(colored, tuple(map(int, corners[i])), tuple(map(int, corners[(i + 1) % 4])), (0, 255, 0),
                     2)
        warped_image = warp_perspective(img, corners)
        warped_image = cv2.flip(warped_image, 0)
        h, w = warped_image.shape[:2]

        if w / h < 0.7:
            # Rotate 90 degrees clockwise if width is less than height
            warped_image = cv2.rotate(warped_image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imshow('Warped Image', warped_image)
        return warped_image
    else:
        return None


def crop_pan(warped_img):
    h, w = warped_img.shape[:2]

    pan_y_start_ratio = 0.45
    pan_y_end_ratio = 0.6
    pan_x_start_ratio = 0.05
    pan_x_end_ratio = 1

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

    _, img_thresh = cv2.threshold(img_straightened, 130, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshold', img_thresh)

    img_resized = cv2.resize(img_thresh, (1000, 700))

    cv2.imshow('img_resized', img_resized)

    pan_image = crop_pan(img_resized)

    return pan_image


input_img = cv2.imread('CSE483 SMR24 Project Test Cases/16 - Sheel el kart yastaaaa.jpg')
pan_image = pre_process(input_img)
if pan_image is not None:
    cv2.imshow('PAN Image', pan_image)
    cv2.imwrite('pan.jpg', pan_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
