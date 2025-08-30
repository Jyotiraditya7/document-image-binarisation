import cv2
import numpy as np

# ---- Load and Preprocess Image ----
img = cv2.imread('images/slant.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
gray_eq = clahe.apply(gray)

# ---- Shadow Removal ----
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
diff = cv2.absdiff(gray_eq, bg)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow("1. Shadow Removed", norm_img)
cv2.waitKey(0)

# ---- Initial Thresholding (Otsu for rough map) ----
_, rough_thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# ---- Find initial contours ----
contours, _ = cv2.findContours(rough_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---- Adaptive Local Thresholding for Each Component ----
formatted = np.zeros_like(gray)
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area < 2000:  # skip tiny noise
        continue

    roi = norm_img[y:y+h, x:x+w]

    # Estimate local stroke width using distance transform
    roi_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 10)
    dist = cv2.distanceTransform(roi_bin, cv2.DIST_L2, 5)
    mean_stroke = int(np.mean(dist[dist > 0]) * 2) | 1  # make it odd

    k = max(15, mean_stroke)
    k = k if k % 2 == 1 else k + 1  # ensure odd

    local_bin = cv2.adaptiveThreshold(roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, k, 10)

    formatted[y:y+h, x:x+w] = local_bin

cv2.imshow("2. Adaptive Local Binarization", formatted)
cv2.waitKey(0)

# ---- Recompute Contours After Adaptive Binarization ----
contours, _ = cv2.findContours(formatted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# ---- Adaptive Morphological Closing ----
adaptive_closed = formatted.copy()
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = w * h
    if area > 3000:
        k_w = max(3, int(w / 10)) | 1
        k_h = max(3, int(h / 4)) | 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, k_h))
        roi = adaptive_closed[y:y+h, x:x+w]
        closed = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel)
        adaptive_closed[y:y+h, x:x+w] = closed

cv2.imshow("3. Adaptive Morph Closing", adaptive_closed)
cv2.waitKey(0)

# ---- Merge Overlapping Contours via Mask ----
contours, _ = cv2.findContours(adaptive_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
merged_boxes = []
padding = 0  # for merging close-by horizontal contours

for i, cnt1 in enumerate(contours):
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    if w1 * h1 < 2000:
        continue

    # expand horizontally only
    box1 = [x1 - padding, y1, x1 + w1 + padding, y1 + h1]
    merged = False

    for j, box2 in enumerate(merged_boxes):
        xa1, ya1, xa2, ya2 = box2
        # Check if horizontal overlap or touch
        if not (box1[2] < xa1 or box1[0] > xa2 or box1[3] < ya1 or box1[1] > ya2):
            new_box = [
                min(xa1, box1[0]),
                min(ya1, box1[1]),
                max(xa2, box1[2]),
                max(ya2, box1[3])
            ]
            merged_boxes[j] = new_box
            merged = True
            break

    if not merged:
        merged_boxes.append(box1)

# ---- Draw Final Merged Boxes ----
output = img.copy()
for box in merged_boxes:
    x1, y1, x2, y2 = box
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 0), 10)

cv2.imshow("4. Final Merged Contours", output)
cv2.waitKey(0)
cv2.destroyAllWindows()