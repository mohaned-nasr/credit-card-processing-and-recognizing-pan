import cv2
import numpy as np


def load_digit_templates():
    templates = {}
    for i in range(10):
        template = cv2.imread(f'templates/{i}.png', cv2.IMREAD_GRAYSCALE)
        if template is not None:
            templates[i] = template
        else:
            print(f"Template for digit {i} not found.")
    return templates


def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(boxes[:, 4])

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        overlap = (w * h) / (area[idxs[:last]])

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")


def match_template(pan_image, templates):
    detected_digits = []

    # Convert the pan_image to color for visualization purposes
    pan_image_color = cv2.cvtColor(pan_image, cv2.COLOR_GRAY2BGR)

    for digit, template in templates.items():
        w, h = template.shape[::-1]

        # Apply template matching
        res = cv2.matchTemplate(pan_image, template, cv2.TM_CCOEFF_NORMED)

        threshold = 0.52  # Adjust this threshold for your specific needs
        loc = np.where(res >= threshold)

        boxes = []
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            top_left = pt
            bottom_right = (top_left[0] + w, top_left[1] + h)
            boxes.append((top_left[0], top_left[1], bottom_right[0], bottom_right[1], res[top_left[1], top_left[0]]))
            cv2.rectangle(pan_image_color, top_left, bottom_right, (0, 255, 0), 1)

        # Apply non-maximum suppression
        boxes = non_max_suppression(boxes, 0.3)

        for (x1, y1, x2, y2, _) in boxes:
            detected_digits.append((x1, digit))

    # Sort digits based on their x-coordinate (left to right)
    detected_digits = sorted(detected_digits, key=lambda x: x[0])

    return pan_image_color, [digit for _, digit in detected_digits]


# Load and preprocess the cropped PAN image
pan_image = cv2.imread('pan.jpg', 0)  # Ensure it's read as grayscale

# Load digit templates
templates = load_digit_templates()

# Perform template matching to detect digits
annotated_image, detected_digits = match_template(pan_image, templates)
print("Detected PAN:", "".join(map(str, detected_digits)))

# To visualize the results
cv2.imshow('Detected Digits', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
