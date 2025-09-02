import cv2
import numpy as np

# ---- Load Image ----
img = cv2.imread("images/10388.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---- Step 1: Adaptive Threshold ----
thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV, 35, 15
)

cv2.imshow("Step1 - Adaptive Threshold", thresh)
cv2.waitKey(0)

# ---- Step 2: Noise Removal (morph open) ----
kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)

cv2.imshow("Step2 - After Morph Open", clean)
cv2.waitKey(0)

# ---- Step 3: Word Grouping (dilate) ----
kernel_word = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 15))
morphed = cv2.dilate(clean, kernel_word, iterations=1)

cv2.imshow("Step3 - After Word Dilation", morphed)
cv2.waitKey(0)

# ---- Step 4: Contours ----
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
for cnt in contours:
    area = cv2.contourArea(cnt)
    if area < 2000:   # 🔑 filter out small noise (adjust threshold as needed)
        continue

    hull = cv2.convexHull(cnt)
    cv2.drawContours(output, [hull], -1, (0, 0, 0), 10)

cv2.imshow("Step4 - Final Word Segmentation", output)
cv2.waitKey(0)
cv2.destroyAllWindows()