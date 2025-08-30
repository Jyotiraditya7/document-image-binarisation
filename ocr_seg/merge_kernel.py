import cv2
import numpy as np
from sklearn.cluster import DBSCAN

# Load and preprocess
img = cv2.imread('images/slant.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray_eq = clahe.apply(gray)

# Shadow removal
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25))
bg = cv2.morphologyEx(gray_eq, cv2.MORPH_CLOSE, kernel)
diff = cv2.absdiff(gray_eq, bg)
norm_img = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

# Binarization
_, light_thresh = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


# Morph closing to connect characters
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (50,2))
closed = cv2.morphologyEx(light_thresh, cv2.MORPH_CLOSE, kernel_close)


# Find contours
contours, _ = cv2.findContours(light_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract bounding boxes
boxes = []
centers_y = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    if w * h > 3000:  # ignore small noise
        boxes.append([x, y, x + w, y + h])
        centers_y.append([y + h // 2])  # Use Y center only for vertical clustering

# Apply DBSCAN to cluster boxes into rows
db = DBSCAN(eps=40, min_samples=1).fit(centers_y)
labels = db.labels_

# Group boxes by cluster
clustered_boxes = {}
for label, box in zip(labels, boxes):
    clustered_boxes.setdefault(label, []).append(box)

# Merge boxes horizontally within each row
merged_rows = []
for box_group in clustered_boxes.values():
    # Sort left-to-right
    box_group = sorted(box_group, key=lambda b: b[0])
    x1 = min([b[0] for b in box_group])
    y1 = min([b[1] for b in box_group])
    x2 = max([b[2] for b in box_group])
    y2 = max([b[3] for b in box_group])
    merged_rows.append((x1, y1, x2, y2))

# Draw final merged line boxes
output = img.copy()
for (x1, y1, x2, y2) in merged_rows:
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 4)

cv2.imshow("DBSCAN Line Detection", output)
cv2.waitKey(0)
cv2.destroyAllWindows()