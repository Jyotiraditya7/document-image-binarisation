import cv2
import numpy as np

img = cv2.imread('images/shadow_bottom.jpg')

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

L_eq = clahe.apply(L)
lab_eq = cv2.merge([L_eq, A, B])

img_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


gray = cv2.cvtColor(img_eq, cv2.COLOR_BGR2GRAY)

cv2.imshow("1. Grayscale Image", gray)
cv2.waitKey(0)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)


norm = cv2.divide(gray, bg, scale=255)

cv2.imshow("2. Normalized Image", norm)
cv2.waitKey(0)

binary = cv2.adaptiveThreshold(
    norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY, 35, 15
)


binary_inv = cv2.bitwise_not(binary)

cv2.imshow("3. Inverted Binary Image", binary_inv)
cv2.waitKey(0)

kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel_small)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
min_area = 500
refined = cleaned.copy()
for i in range(1, num_labels):  
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        refined[labels == i] = 0

cv2.imshow("Final Shadow Removed (White Text)", refined)
cv2.waitKey(0)
cv2.destroyAllWindows()
