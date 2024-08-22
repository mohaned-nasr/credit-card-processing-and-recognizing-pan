from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os


def create_digit_template(digit, font_path, font_size=60, image_size=(48, 96), thickness="regular"):
    # Create a blank white image using PIL
    image = Image.new('L', image_size, color=0)
    draw = ImageDraw.Draw(image)

    # Load the custom font
    font = ImageFont.truetype(font_path, font_size)

    # Get the bounding box of the text and calculate the position to center it
    bbox = draw.textbbox((0, 0), str(digit), font=font)
    text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2

    # Draw the digit on the image
    draw.text((text_x, text_y), str(digit), font=font, fill=255)

    # Convert the PIL image to a NumPy array (compatible with OpenCV)
    template = np.array(image)

    if thickness == "thin":
        # Apply morphological erosion to create a thin template
        kernel = np.ones((2, 3), np.uint8)
        template = cv2.erode(template, kernel, iterations=1)

    return template


def generate_digit_templates(font_path):
    if not os.path.exists('templates/regular'):
        os.makedirs('templates/regular')
    if not os.path.exists('templates/thin'):
        os.makedirs('templates/thin')

    templates = {"regular": {}, "thin": {}}

    for i in range(10):
        # Generate regular template
        regular_template = create_digit_template(i, font_path, thickness="regular")
        templates["regular"][i] = regular_template
        cv2.imwrite(f'templates/regular/{i}.png', regular_template)  # Save each regular template as an image file

        # Generate thin template
        thin_template = create_digit_template(i, font_path, thickness="thin")
        templates["thin"][i] = thin_template
        cv2.imwrite(f'templates/thin/{i}.png', thin_template)  # Save each thin template as an image file

    return templates


# Example usage
font_path = "CREDIT CARD/CREDC___.ttf"  # Replace with the path to your .ttf or .otf file
digit_templates = generate_digit_templates(font_path)

# Display the regular and thin templates for digit 0
cv2.imshow("Digit 0 Regular Template", digit_templates["regular"][0])
cv2.imshow("Digit 0 Thin Template", digit_templates["thin"][0])
cv2.waitKey(0)
cv2.destroyAllWindows()
