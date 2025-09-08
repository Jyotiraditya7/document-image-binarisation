import cv2
import numpy as np

img = cv2.imread('images/shadow_bottom.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                # convert to grayscale

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # makes text strokes more visible
gray_eq = clahe.apply(gray)
cv2.imshow("0",gray_eq)
cv2.waitKey(0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))    # Morphological closing (dilation + erosion)
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
diff = cv2.absdiff(gray_eq, bg)                                     # removes shadows, leaving mostly text strokes.

norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

cv2.imshow("1. Shadow Removed", norm_img)
cv2.waitKey(0)

blurred = cv2.GaussianBlur(norm_img, (5,5), 0)

cv2.imshow("2. Blurred Image", blurred)
cv2.waitKey(0)

_, binarized_img = cv2.threshold(
    norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imshow("Binarized Image (Global)", binarized_img)
cv2.waitKey(0)


num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binarized_img, connectivity=8)

# keep only large components
min_area = 1000  
refined = binarized_img.copy()

for i in range(1, num_labels):  # skip background
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        refined[labels == i] = 0 


cv2.imshow("Filtered Large Components", refined)
cv2.waitKey(0)


shape = "rect"      
ksize = (15, 10)      
iterations = 1          


shape_map = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS
}


kernel = cv2.getStructuringElement(shape_map[shape], ksize)


dilated = cv2.dilate(refined, kernel, iterations=iterations)


closed_manual = cv2.erode(dilated, kernel, iterations=iterations)

cv2.imshow("Manual Closing", closed_manual)
cv2.waitKey(0)


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_manual, connectivity=8)

output = cv2.cvtColor(closed_manual, cv2.COLOR_GRAY2BGR)

min_area = 1000
word_id = 0

for i in range(1, num_labels):
    x, y, w, h, area = stats[i]
    if area >= min_area:
        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        word_crop = closed_manual[y:y+h, x:x+w]
        word_id += 1

cv2.imshow("Word Boxes (Connected Components)", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

