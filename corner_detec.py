import cv2
import numpy as np

# Load the image
img = cv2.imread('i.jpg')  # Replace with your image path

# Check if image loaded successfully
if img is None:
    print("Error: Could not load image. Check the file path and format.")
else:
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create binary image (invert if black is foreground)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

    # Apply Canny edge detection to find the edges of the black line
    edges = cv2.Canny(binary, 50, 150)

    # Detect corners using Shi-Tomasi method on the edges
    corners = cv2.goodFeaturesToTrack(edges, maxCorners=4, qualityLevel=0.01, minDistance=50)
    corners = np.int0(corners)

    # Draw corners on the original image
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    # Display the result
    cv2.imshow('Corners on Edge', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()