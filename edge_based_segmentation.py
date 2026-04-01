import cv2
import numpy as np
from google.colab.patches import cv2_imshow

# Load image
img = cv2.imread('/content/zebra.jpg', 0)

# Check if image loaded
if img is None:
    print("Error: Image not found")
else:
    # Resize
    img = cv2.resize(img, (400, 400))

    # -------------------------------
    # 1. Sobel (Gradient)
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(sobel / np.max(sobel) * 255)

    # -------------------------------
    # 2. Laplacian
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # -------------------------------
    # 3. Canny
    canny = cv2.Canny(img, 100, 200)

    # -------------------------------
    # 4. Edge Linking (Hough Lines)
    edges = cv2.Canny(img, 100, 200)
    hough_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Tuned parameters (important for zebra)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 150,
                            minLineLength=100, maxLineGap=20)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # -------------------------------
    # Convert grayscale to color
    def to_color(image):
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    img_c = to_color(img)
    sobel_c = to_color(sobel)
    lap_c = to_color(laplacian)
    canny_c = to_color(canny)

    # -------------------------------
    # Add labels
    def put_text(image, text):
        img_copy = image.copy()
        cv2.putText(img_copy, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0), 2)
        return img_copy

    img_c = put_text(img_c, "Original")
    sobel_c = put_text(sobel_c, "Gradient Method (Sobel)")
    lap_c = put_text(lap_c, "Second Order(Laplacian)")
    canny_c = put_text(canny_c, "Canny")
    hough_img = put_text(hough_img, "Edge Linking")

    # -------------------------------
    # Create blank image (to avoid duplication)
    blank = np.zeros_like(img_c)
    blank = put_text(blank, " ")

    # -------------------------------
    # Combine into 2x3 grid
    top = np.hstack((img_c, sobel_c, lap_c))
    bottom = np.hstack((canny_c, hough_img, blank))
    combined = np.vstack((top, bottom))

    # -------------------------------
    # Show output
    cv2_imshow(combined)
