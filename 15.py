import numpy as np
import cv2


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

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped


def color_correct(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_corrected = clahe.apply(img_gray)

    _, img_thresh = cv2.threshold(img_corrected, np.median(img_corrected), 255, cv2.THRESH_BINARY)

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

    cv2.imshow('edges', edges)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        print("No lines detected.")
        return None

    extended_lines = []

    for rho, theta in lines[:, 0]:
        print(rho, theta)
        # # a=math.cos(theta)
        # rounded = round(a,6)
        # print(rounded)
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * -b)
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * -b)
        y2 = int(y0 - 1000 * a)

        print(x1, y1, x2, y2)

        (x1_ext, y1_ext), (x2_ext, y2_ext) = extend_line(x1, y1, x2, y2, img_resized.shape[:2])
        if (x1_ext, y1_ext) is not None and (x2_ext, y2_ext) is not None:
            extended_lines.append(((x1_ext, y1_ext), (x2_ext, y2_ext)))
            cv2.line(img_lines, (x1_ext, y1_ext), (x2_ext, y2_ext), (0, 0, 255), 2)
            # plt.imshow(img_lines)
            cv2.imshow('lines', img_lines)
            # plt.show()

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

        cv2.imshow('Detected Card Frame', img_lines)
        cv2.imshow('Warped Image', warped_image)
        return warped_image
    else:
        return None


def crop_pan(warped_img):
    h, w = warped_img.shape[:2]

    pan_y_start_ratio = 0.5
    pan_y_end_ratio = 0.7
    pan_x_start_ratio = 0.08
    pan_x_end_ratio = 0.9

    pan_x_start = int(pan_x_start_ratio * w)
    pan_x_end = int(pan_x_end_ratio * w)
    pan_y_start = int(pan_y_start_ratio * h)
    pan_y_end = int(pan_y_end_ratio * h)

    pan_region = warped_img[pan_y_start:pan_y_end, pan_x_start:pan_x_end]

    return pan_region


def pre_process(img):
    img_colored = img.copy()
    img_gray = color_correct(img)
    img_noise_removed = remove_periodic_noise(img_gray)
    img_straightened = straighten(img_noise_removed, img_colored)

    if img_straightened is None:
        print("No card detected.")
        return None

    _, img_thresh = cv2.threshold(img_straightened, 160, 255, cv2.THRESH_BINARY)

    cv2.imshow('threshold', img_thresh)

    img_resized = cv2.resize(img_thresh, (1000, 700))

    cv2.imshow('img_resized', img_resized)

    pan_image = crop_pan(img_resized)

    return pan_image


# Example usage:
input_img = cv2.imread('CSE483 SMR24 Project Test Cases/15 - Compresso Espresso.jpg')
pan_image = pre_process(input_img)
if pan_image is not None:
    cv2.imshow('PAN Image', pan_image)
    cv2.imwrite('pan.jpg', pan_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
