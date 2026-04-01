import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import cv2_imshow for Colab

# Step 1: Load image
img = cv2.imread('/content/zebra.jpg')

# Check if image was loaded successfully
if img is None:
    print('Error: Could not load image. Please check the path.')
else:
    # Step 2: Convert to smaller (odd size) to exaggerate effect
    small = cv2.resize(img, (101, 101))   # odd size

    # Step 3: Apply interpolation methods
    nearest = cv2.resize(small, (400, 400), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(small, (400, 400), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(small, (400, 400), interpolation=cv2.INTER_CUBIC)

    # Step 4: Resize original for display
    original = cv2.resize(img, (400, 400))

    # Step 5: Add titles on images
    def put_text(image, text):
        img_copy = image.copy()
        cv2.putText(img_copy, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        return img_copy

    original = put_text(original, "Original")
    nearest = put_text(nearest, "Nearest Neighbor")
    bilinear = put_text(bilinear, "Bilinear")
    bicubic = put_text(bicubic, "Bicubic")

    # Step 6: Combine images (2x2 grid)
    top = np.hstack((original, nearest))
    bottom = np.hstack((bilinear, bicubic))
    combined = np.vstack((top, bottom))

    # Step 7: Show result using cv2_imshow for Colab
    cv2_imshow(combined)
