import cv2
import numpy as np


img = cv2.imread('images/shadow_side.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)

diff = cv2.absdiff(gray_eq, bg)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("Shadow Removed", norm_img)
cv2.waitKey(0)


_, light_thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Light Threshold (Otsu)", light_thresh)
cv2.waitKey(0)


contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

output = img.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 3000:  # Ignore small patches
        cv2.rectangle(output, (x, y), (x + w, y + h), (15, 255, 80), 10)

cv2.imshow("Final Contours on Original", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
