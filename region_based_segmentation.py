import cv2
import numpy as np
from google.colab.patches import cv2_imshow # Import cv2_imshow for Colab

# Load image (grayscale)
img = cv2.imread('/content/zebra.jpg', 0)

# Check if image was loaded successfully
if img is None:
    print('Error: Could not load image. Please check the path.')
else:
    img = cv2.resize(img, (400, 400))

    # -------------------------------
    # 1. Thresholding
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # -------------------------------
    # 2. Region Growing
    def region_growing(img, seed):
        h, w = img.shape
        visited = np.zeros((h, w), dtype=bool)
        output = np.zeros((h, w), np.uint8)

        stack = [seed]
        threshold = 10
        seed_value = img[seed]

        while stack:
            x, y = stack.pop()

            if visited[x, y]:
                continue

            visited[x, y] = True

            if abs(int(img[x, y]) - int(seed_value)) < threshold:
                output[x, y] = 255

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w:
                            stack.append((nx, ny))

        return output

    region = region_growing(img, (200, 200))

    # -------------------------------
    # 3. Region Merging (Connected Components)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels = cv2.connectedComponents(binary)

    # Convert labels to displayable image
    labels_img = np.uint8(255 * labels / np.max(labels))

    # -------------------------------
    # Convert to color for labeling
    def to_color(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img_c = to_color(img)
    thresh_c = to_color(thresh)
    region_c = to_color(region)
    labels_c = to_color(labels_img)

    # -------------------------------
    # Add text
    def put_text(image, text):
        img_copy = image.copy()
        cv2.putText(img_copy, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        return img_copy

    img_c = put_text(img_c, "Original")
    thresh_c = put_text(thresh_c, "Threshold")
    region_c = put_text(region_c, "Region Growing")
    labels_c = put_text(labels_c, "Region Merging")

    # -------------------------------
    # Combine (2x2)
    top = np.hstack((img_c, thresh_c))
    bottom = np.hstack((region_c, labels_c))
    combined = np.vstack((top, bottom))

    # -------------------------------
    cv2_imshow(combined)
